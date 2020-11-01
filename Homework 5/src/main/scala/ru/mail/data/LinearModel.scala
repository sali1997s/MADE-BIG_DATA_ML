package ru.mail.data

import breeze.linalg.{DenseMatrix, DenseVector}

case class LinearModel(w: DenseVector[Double]){
  def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    X * w
  }
}
