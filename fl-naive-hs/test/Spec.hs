import Test.Hspec
import Numeric.LinearAlgebra
import Lib

testBellman :: IO ()
testBellman = hspec $ do
  let q = konst 0 (16, 4) :: Matrix R
  let q1 = accum q (+) [((0, 0), 0.8)]
  let q2 = accum q (+) [((0, 1), 0.8)]
  let q3 = accum q (+) [((0, 0), 0.8), ((0, 1), 0.8)]
  let q4 = accum q (+) [((0, 1), 0.6080000000000001), ((1, 0), 0.8)]
  describe "Bellman equation" $ do
    it "With no reward matrix not updated" $ do
      updateMatrixBellman q 0 0 0.8 0 0.95 1 `shouldBe` q
    it "With reward matrix updated" $ do
      updateMatrixBellman q 0 0 0.8 1 0.95 1 `shouldBe` q1
    it "Update non zero matrix" $ do
      updateMatrixBellman q2 0 0 0.8 1 0.95 1 `shouldBe` q3
    it "Double update" $ do
      let first = updateMatrixBellman q 1 0 0.8 1.0 0.95 15
      updateMatrixBellman first 0 1 0.8 0.0 0.95 1 `shouldBe` q4

main :: IO ()
main = testBellman
