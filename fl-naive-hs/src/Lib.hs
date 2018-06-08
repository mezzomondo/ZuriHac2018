{-# LANGUAGE NamedFieldPuns    #-}
{-# LANGUAGE OverloadedStrings #-}

module Lib
    ( frozenLakeMain
    , updateMatrixBellman
    ) where

import Control.Exception.Base
-- import Control.Lens
import Control.Monad
import Control.Monad.Catch
import Control.Monad.IO.Class
-- import Data.Aeson
import Data.Aeson.Types
import Data.Scientific
import Data.Text
import Network.HTTP.Client
import Numeric.LinearAlgebra
import OpenAI.Gym
import Servant.Client
import System.Random

{-data Env = Env {
    learningRate :: Double
  , gamma        :: Double
  , numEpisodes  :: Int
  , qTable       :: Matrix R
}-}

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
    Right ok -> print ok
  where
    url :: BaseUrl
    url = BaseUrl Http "localhost" 5000 ""

argMax :: Matrix R -> Matrix R -> IO Int
argMax row noise = do
  seed <- randomIO
  let dim = cols row
  let newRow = row + asRow (randomVector seed Gaussian dim) * noise
  return $ snd $ maxIndex newRow

updateMatrixBellman :: Matrix R -> Int -> Int -> Double -> Double -> Double -> Int -> Matrix R
updateMatrixBellman qTable s a lr reward y s1 =
  accum qTable (+) [((s, a), bellman)]
  where
    bellman = lr * (reward + y * max) - atIndex qTable (s, a)
    max = maxElement (qTable ? [s1])

frozenLake :: ClientM [Matrix R]
frozenLake = do
  inst <- envCreate FrozenLakeV0
  replicateM episodeCount (agent inst)
  where
    episodeCount :: Int
    episodeCount = 100

agent :: InstID -> ClientM (Matrix R)
agent inst = do
  Observation obS <- envReset inst
  let s = valueToInt obS
  osInfo <- envObservationSpaceInfo inst
  let osN = getDimension osInfo
  asInfo <- envActionSpaceInfo inst
  let asN = getDimension asInfo
  let q = konst 0 (osN, asN) :: Matrix R
  let learningRate = 0.8
  let gamma = 0.95
  go q s 1 learningRate gamma False
  where
    maxSteps :: Int
    maxSteps = 100

    go :: Matrix R -> Int -> Int -> Double -> Double -> Bool -> ClientM (Matrix R)
    go q s x lr g done = do
      let row = q ? [s]
      nextStep <- liftIO $ argMax row (1.0/fromIntegral x)
      Outcome ob reward done _ <- envStep inst (Step (Number (fromIntegral nextStep)) True)
      if not done && x < maxSteps
        then do
            let s1 = valueToInt ob
            let q1 = updateMatrixBellman q s nextStep lr reward g s1
            if reward > 0.0
              then
                liftIO $ print q1
              else
                liftIO $ print "NONONO"
              
            go q1 s1 (x + 1) lr g done
        else
            return q
