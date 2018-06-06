{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE OverloadedLists       #-}

module Lib
    ( runCalc
    ) where

import Control.Monad (replicateM_)
import Control.Monad.IO.Class (liftIO)
import qualified Data.Vector as V
import Data.Vector (Vector)
import TensorFlow.Core
  (Tensor, Value, feed, encodeTensorData, Scalar(..), ControlNode, Build, TensorData)
import TensorFlow.Ops
  (add, placeholder, sub, reduceSum, mul, save, restore)
import TensorFlow.GenOps.Core (square)
import TensorFlow.Variable (readValue, initializedVariable, Variable)
import TensorFlow.Session (runSession, run, runWithFeeds, build, Session)
import TensorFlow.Minimize (gradientDescent, minimizeWith)
import TensorFlow.Tensor

data Model = Model {
      train :: TensorData Float -> TensorData Float -> Session ()
    , getResults :: Session (Scalar Float, Scalar Float)
}

createModel :: Vector Float -> Vector Float -> Build Model
createModel xIn yIn = do
    let xSize = fromIntegral $ V.length xIn
    let ySize = fromIntegral $ V.length yIn

    (w :: Variable Float) <- initializedVariable 3
    (b :: Variable Float) <- initializedVariable 1

    (x :: Tensor Value Float) <- placeholder [xSize]
    (y :: Tensor Value Float) <- placeholder [ySize]

    let linearModel = (readValue w `mul` x) `add` readValue b

    let squareDeltas = square (linearModel `sub` y)

    let loss = reduceSum squareDeltas
    tStep <- minimizeWith (gradientDescent 0.01) loss [w, b]

    return Model {
        train = \xF yF -> runWithFeeds [feed x xF, feed y yF] tStep
      , getResults = run (readValue w, readValue b)
    }

runCalc :: IO ()
runCalc = runSession $ do
    let xInput = V.fromList [1.0, 2.0, 3.0, 4.0]
    let yInput = V.fromList [4.0, 9.0, 14.0, 19.0]
    let xSize = fromIntegral $ V.length xInput
    let ySize = fromIntegral $ V.length yInput

    -- Create the model
    model <- build $ createModel xInput yInput

    -- Train
    replicateM_ 1000 $
        train model (encodeTensorData [xSize] xInput) (encodeTensorData [ySize] yInput) 

    -- Get training results
    (w_learned, b_learned) <- getResults model

    liftIO $ print (unScalar w_learned, unScalar b_learned)
