{-# LANGUAGE NamedFieldPuns    #-}
{-# LANGUAGE OverloadedStrings #-}

module Lib
    ( frozenLakeMain
    , updateMatrixBellman
    , Env (..)
    ) where

import System.Console.ANSI
import Control.Exception.Base
-- import Control.Lens
import Data.IORef
import Control.Monad
import Control.Monad.Catch
import Control.Monad.IO.Class
import Control.Monad.Reader
import Control.Monad.ST
-- import Data.Aeson
import Data.Aeson.Types
import Data.Scientific
import Data.Text
import Network.HTTP.Client
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel
import OpenAI.Gym
import Servant.Client
import Text.Printf
import System.Random

data Env = Env {
    envInstId       :: InstID
  , envLearningRate :: Double
  , envGamma        :: Double
  , envNumEpisodes  :: Int
  , envQTable       :: IORef (Matrix R)
  , envRewards      :: IORef [Double]
}

type App = ReaderT Env ClientM

{- REMEMBER:
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
-}

data MyInfo = MyInfo {
    n    :: Int
  , name :: String
} deriving (Show)

getDimension :: Info -> Int
getDimension info = n myInfo
  where
    Just myInfo = parseMaybe parseInfo info
    parseInfo :: Info -> Parser MyInfo
    parseInfo (Info i) =
      MyInfo <$> i .: "n"
        <*> i .: "name"

valueToInt :: Value -> Int
valueToInt (Number sc) = finalInt
  where
    Just finalInt = toBoundedInteger sc :: Maybe Int

argMax :: Matrix R -> Matrix R -> IO Int
argMax row noise = do
  seed <- randomIO
  let dim = cols row
  let newRow = row + asRow (randomVector seed Gaussian dim) * noise
  return $ snd $ maxIndex newRow

updateMatrixBellman :: Env -> Int -> Int -> Double -> Int -> IO (Matrix R)
updateMatrixBellman env currentState currentAction reward nextState = do
  qTable <- readIORef $ envQTable env
  let lr = envLearningRate env
  let g = envGamma env
  let max = maxElement (qTable ? [nextState])
  let bellman = curValue + lr * ((reward + g * max) - curValue)
        where
          curValue = atIndex qTable (currentState, currentAction)
  return $ updateMatrix qTable currentState currentAction bellman

updateMatrix :: Matrix R -> Int -> Int -> Double -> Matrix R
updateMatrix q s a b = runST $ do
  mutableQ <- thawMatrix q
  writeMatrix mutableQ s a b
  freezeMatrix mutableQ

buildEnv :: ClientM Env
buildEnv = do
  inst <- envCreate FrozenLakeV0
  let learningRate = 0.8
  let gamma = 0.95
  let numEpisodes = 500
  osInfo <- envObservationSpaceInfo inst
  let osN = getDimension osInfo
  asInfo <- envActionSpaceInfo inst
  let asN = getDimension asInfo
  ref <- liftIO $ newIORef (konst 0 (osN, asN))
  rewardsRef <- liftIO $ newIORef []
  return Env {
      envInstId       = inst
    , envLearningRate = learningRate
    , envGamma        = gamma
    , envNumEpisodes  = numEpisodes
    , envQTable       = ref
    , envRewards      = rewardsRef
  }

compass :: Int -> Int -> Int
compass original final = case final - original of
  (-1) -> 0
  4    -> 1
  1    -> 2
  (-4) -> 3
  0    -> -1
  _    -> -100

trueAction :: Int -> Int -> Int
trueAction original computed = if computed == -1 then original else computed

printMatrix :: Matrix R -> IO ()
printMatrix q = do
  putStrLn "LEFT      DOWN      RIGHT     UP      "
  putStr $ format "  " (printf "%.6f") q

frozenLakeMain :: IO ()
frozenLakeMain = do
  manager <- newManager defaultManagerSettings
  out <- runClientM frozenLake (ClientEnv manager url)
  case out of
    Left err -> print err
    Right env -> do
      clearFromCursorToScreenEnd
      q <- readIORef (envQTable env)
      putStrLn "----------- FINAL Q-TABLE ------------"
      printMatrix q
      r <- readIORef (envRewards env)
      putStrLn "---------- SCORE OVER TIME -----------"
      print $ sum r/fromIntegral (envNumEpisodes env)
  where
    url :: BaseUrl
    url = BaseUrl Http "localhost" 5000 ""

frozenLake :: ClientM Env
frozenLake = do
  env <- buildEnv
  runReaderT (replicateM_ (envNumEpisodes env) agent) env
  return env

agent :: App ()
agent = do
  env <- ask
  Observation obS <- lift $ envReset $ envInstId env
  let firstState = valueToInt obS
  go firstState 1 False

go :: Int-> Int -> Bool -> App ()
go state loopStep done = do
  env <- ask
  q <- liftIO $ readIORef $ envQTable env
  let row = q ? [state]
  nextAction <- liftIO $ argMax row (1.0/fromIntegral loopStep)
  Outcome ob reward done _ <- lift $ envStep (envInstId env) (Step (Number (fromIntegral nextAction)) True)
  let nextState = valueToInt ob
  let computedAction = compass state nextState
  when (computedAction >= 0) $ do
    let action = trueAction nextAction computedAction
    newQTable <- liftIO $ updateMatrixBellman env state action reward nextState
    liftIO $ writeIORef (envQTable env) newQTable
    liftIO $ printMatrix newQTable
    liftIO $ cursorUp 17
  if done then do
            rewards <- liftIO $ readIORef (envRewards env)
            liftIO $ writeIORef (envRewards env) (rewards++[reward])
          else go nextState (loopStep + 1) done