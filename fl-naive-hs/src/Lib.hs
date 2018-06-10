{-# LANGUAGE NamedFieldPuns    #-}
{-# LANGUAGE OverloadedStrings #-}

module Lib
    ( frozenLakeMain
    , updateMatrixBellman
    ) where

import Control.Exception.Base
-- import Control.Lens
import Data.IORef
import Control.Monad
import Control.Monad.Catch
import Control.Monad.IO.Class
import Control.Monad.Reader
-- import Data.Aeson
import Data.Aeson.Types
import Data.Scientific
import Data.Text
import Network.HTTP.Client
import Numeric.LinearAlgebra
import OpenAI.Gym
import Servant.Client
import System.Random

data Env = Env {
    envInstId       :: InstID
  , envLearningRate :: Double
  , envGamma        :: Double
  , envNumEpisodes  :: Int
  , envMaxSteps     :: Int
  , envQTable       :: IORef (Matrix R)
}

type App = ReaderT Env ClientM

{- REMEMBER:
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

IS THAT TRUE?????
-}

data MyInfo = MyInfo {
    n    :: Int
  , name :: String
} deriving (Show)

parseInfo :: Info -> Parser MyInfo
parseInfo (Info i) =
  MyInfo <$> i .: "n"
         <*> i .: "name"

getDimension :: Info -> Int
getDimension info = n myInfo
  where
    Just myInfo = parseMaybe parseInfo info

valueToInt :: Value -> Int
valueToInt (Number sc) = finalInt
  where
    Just finalInt = toBoundedInteger sc :: Maybe Int

frozenLakeMain :: IO ()
frozenLakeMain = do
  manager <- newManager defaultManagerSettings
  out <- runClientM frozenLake (ClientEnv manager url)
  case out of
    Left err -> print err
    Right ref -> do
                    q <- readIORef ref
                    putStrLn "---------- FINAL Q-TABLE -----------"
                    disp 6 q
  where
    url :: BaseUrl
    url = BaseUrl Http "localhost" 5000 ""

argMax :: Matrix R -> Matrix R -> IO Int
argMax row noise = do
  seed <- randomIO
  let dim = cols row
  let newRow = row + asRow (randomVector seed Gaussian dim) * noise
  return $ snd $ maxIndex newRow

updateMatrixBellman :: Int -> Int -> Double -> Int -> App (Matrix R)
updateMatrixBellman currentState currentAction reward nextState = do
  env <- ask
  qTable <- liftIO $ readIORef $ envQTable env
  let lr = envLearningRate env
  let g = envGamma env
  let max = maxElement (qTable ? [nextState])
  let bellman = lr * ((reward + g * max) - atIndex qTable (currentState, currentAction))
  let newTable = accum qTable (+) [((currentState, currentAction), bellman)]
  return newTable

buildEnv :: ClientM Env
buildEnv = do
  inst <- envCreate FrozenLakeV0
  let learningRate = 0.8
  let gamma = 0.95
  let numEpisodes = 2000
  let maxSteps = 100
  osInfo <- envObservationSpaceInfo inst
  let osN = getDimension osInfo
  asInfo <- envActionSpaceInfo inst
  let asN = getDimension asInfo
  ref <- liftIO $ newIORef (konst 0 (osN, asN))
  return Env {
      envInstId       = inst
    , envLearningRate = learningRate
    , envGamma        = gamma
    , envNumEpisodes  = numEpisodes
    , envMaxSteps     = maxSteps
    , envQTable       = ref
  }

frozenLake :: ClientM (IORef (Matrix R))
frozenLake = do
  env <- buildEnv
  runReaderT (replicateM_ (envNumEpisodes env) agent) env
  return $ envQTable env

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
  nextStep <- liftIO $ argMax row (1.0/fromIntegral loopStep)
  Outcome ob reward done _ <- lift $ envStep (envInstId env) (Step (Number (fromIntegral nextStep)) True)
  let nextState = valueToInt ob
  newQTable <- updateMatrixBellman state nextStep reward nextState
  liftIO $ writeIORef (envQTable env) newQTable
  liftIO $ disp 6 newQTable
  when (not done && loopStep < envMaxSteps env) $ go nextState (loopStep + 1) done
