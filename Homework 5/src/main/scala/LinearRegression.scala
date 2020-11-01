import java.io.File
import java.io.PrintWriter

import ru.mail.data.LinearModel
import breeze.linalg.{DenseMatrix, DenseVector, csvread, csvwrite, inv, pinv, sum}
import breeze.numerics.{abs, pow}
import breeze.stats.mean


object LinearRegression {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("worng args")
    }
    val file = new File(args(0))
    val fileX = csvread(file, separator = ',',skipLines = 1)
    val X = fileX(::,0 to (fileX.cols - 2))
    val y = fileX(::,fileX.cols - 1)
    val w = inv(X.t * X) * X.t * y
    val model = LinearModel(w)
    val testFile = new File(args(1))
    val fileX_test = csvread(testFile, separator = ',',skipLines = 1)
    val X_test = fileX_test(::,0 to (fileX_test.cols - 2))
    val y_test = fileX_test(::,fileX_test.cols - 1)
    val prediction = model.predict(X_test)
    val out_file = new File("predictions.txt")
    val score_file = new PrintWriter(new File("score.txt"))
    csvwrite(out_file, prediction.toDenseMatrix, ',')
    score_file.printf("MSE: %f, MAE: %f", mean((y_test - prediction) * (y_test - prediction)), mean(abs(y_test - prediction)))
    score_file.close()
  }
}
