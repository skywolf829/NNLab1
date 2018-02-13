
static class Perceptron{
  static Matrix Activate(Matrix m){
    for(int i = 0; i < m.getRowDimension(); i++){
     for(int j = 0; j < m.getColumnDimension(); j++){
       if(m.get(i, j) >= 0) m.set(i, j, 1);
       else m.set(i, j, 0);
     }
    }
    return m;
  }
}

static class MPneuron{  
  static Matrix Activate(Matrix m){
    for(int i = 0; i < m.getRowDimension(); i++){
     for(int j = 0; j < m.getColumnDimension(); j++){
       if(m.get(i, j) >= 0) m.set(i, j, 1);
       else m.set(i, j, -1);
     }
    }
    return m;
  }  
}

static class SigmoidNeuron{  
  static Matrix Activate(Matrix m){
    for(int i = 0; i < m.getRowDimension(); i++){
     for(int j = 0; j < m.getColumnDimension(); j++){
       m.set(i, j, 1.0 / (1 + Math.exp(-m.get(i, j))));
     }
    }
    return m;
  }  
  static Matrix ActivationPrime(Matrix m){
    m = Activate(m);
    for(int i = 0; i < m.getRowDimension(); i++){
      for(int j = 0; j < m.getColumnDimension(); j++){
        m.set(i, j, 
          m.get(i, j) * (1 - m.get(i, j)));
      }
    }
    /*
    for(int i = 0; i < m.getRowDimension(); i++){
      for(int j = 0; j < m.getColumnDimension(); j++){
        m.set(i, j, 
        (Math.exp(-m.get(i, j))) / (Math.pow(1 + Math.exp(-m.get(i, j)), 2))
        );
      }
    }
    */
    return m;
  }
}