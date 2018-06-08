import Test.Hspec
import Numeric.LinearAlgebra
import Lib

testBellman :: IO ()
testBellman = hspec $ do
  let q = konst 0 (16, 4) :: Matrix R
  describe "Bellman equation" $ do
    it "With no reward matrix not updated" $ do
      updateMatrixBellman q 0 0 0.8 0 0.95 1 `shouldBe` q
    it "With reward matrix updated" $ do
      updateMatrixBellman q 0 0 0.8 1 0.95 1 `shouldBe` (accum q (+) [((0, 0), 0.8)])

main :: IO ()
main = testBellman
