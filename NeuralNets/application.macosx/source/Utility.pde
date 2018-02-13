static float distance(Point p1, Point p2) {
  return pow(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2), 0.5);
}
static float distance(float x1, float y1, float x2, float y2) {
  return pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5);
}
static float nChoosek(int n, int k) {
  return (float)fact(n) / (float)(fact(k) * fact(n - k));
}
static final int fact(int num) {
  if (num == 0) return 1;
  return num == 1 ? 1 : fact(num - 1)*num;
}
static Matrix ElementwiseMultiplication(Matrix m1, Matrix m2) {
  Matrix m = new Matrix(m1.getRowDimension(), m1.getColumnDimension());
  for (int i = 0; i < m.getRowDimension(); i++) {
    for (int j = 0; j < m.getColumnDimension(); j++) {
      m.set(i, j, m1.get(i, j) * m2.get(i, j));
    }
  }
  return m;
}
public static boolean isInteger(String s) {
  boolean isValidInteger = false;
  try
  {
    Integer.parseInt(s);

    // s is a valid integer

    isValidInteger = true;
  }
  catch (NumberFormatException ex)
  {
    // s is not an integer
  }

  return isValidInteger;
}
public static void printMatrix(Matrix m) {
  for (int i = 0; i < m.getRowDimension(); i++) {
    for (int j = 0; j < m.getColumnDimension(); j++) {
      print(m.get(i, j) + " ");
    }
    println();
  }
  println();
  println();
}