
import Jama.*;
import Jama.examples.*;
import Jama.test.*;
import Jama.util.*;

int screenWidth = 900;
int screenHeight = 500;
int iteration = 0;
double biggestErrorForEpoch = 0;

SigmoidNetwork nn;
TextInput numInputs, numOutputs, numLayers, nodesPerLayer;
IntSlider speed;
ArrayList<TextInput> inputs = new ArrayList<TextInput>();
Rectangle init;
Rectangle begin;

void start() {

  numInputs = new TextInput(5, 50, 50, 25);
  numOutputs = new TextInput(5, 105, 50, 25);
  numLayers = new TextInput(5, 155, 50, 25);
  nodesPerLayer = new TextInput(5, 205, 50, 25);
  init = new Rectangle(5, 280, 130, 30);
  init.c = color(255);
  begin = new Rectangle(150, 280, 130, 30);
  begin.c = color(255);
  speed = new IntSlider("Speed", 0, 2000, 20, 430, 200);
  nn = new SigmoidNetwork();
  nn.x = 340;
  nn.y = 5;
}
void setup() {
  size(900, 500);
  noSmooth();
  background(255);
}

void draw() {
  background(150);
  stroke(255);


  fill(0);
  textSize(20);
  text("Number of inputs", 5, 25 + 20);
  numInputs.draw();
  fill(0);
  textSize(20);
  text("Number of outputs", 5, 80 + 20);
  numOutputs.draw();
  fill(0);
  textSize(20);
  text("Number of hidden layers", 5, 130 + 20);
  numLayers.draw();
  fill(0);
  textSize(20);
  text("Nodes per hidden layer", 5, 180 + 20);
  nodesPerLayer.draw();

  init.draw();
  fill(0);
  textSize(22);
  text("Initialize", 20, 305);
  if (nn.initialized) {
    begin.draw();
    fill(0);
    textSize(22);
    if (nn.training) text("Stop", 190, 305);
    else text("Begin", 190, 305);
  }
  textSize(16);
  text("Current cost: " + nn.currentCost, 20, 350);
  text("Largest cost this epoch: " + biggestErrorForEpoch, 20, 380);
  text("Epoch " + iteration / 16 + ", iteration " + iteration, 20, 410);
  speed.draw();
  nn.draw();
}
void mousePressed() {
  numInputs.mousePressed();
  numOutputs.mousePressed();
  numLayers.mousePressed();
  nodesPerLayer.mousePressed();
  speed.mousePressed();
  for (int i = 0; i < inputs.size(); i++) {
    inputs.get(i).mousePressed();
  }
  if (init.contains(mouseX, mouseY) && !nn.training) {
    iteration = 0;
    nn.SetInputSize(Integer.parseInt(numInputs.text));
    nn.SetOutputSize(Integer.parseInt(numOutputs.text));
    nn.SetNumLayers(Integer.parseInt(numLayers.text));
    int[] layerSizes = new int[Integer.parseInt(numLayers.text)];
    int temp = 0;
    int spot = 0;
    if (Integer.parseInt(numLayers.text) > 0) {
      for (int i = 0; i < nodesPerLayer.text.length(); i++) {
        if (isInteger(nodesPerLayer.text.substring(i, i+1))) {
          temp *= 10;
          temp += Integer.parseInt(nodesPerLayer.text.substring(i, i+1));
        } else {
          layerSizes[spot] = temp;
          spot++;
          temp = 0;
        }
      }
      layerSizes[spot] = temp;
      nn.SetNodesInEachLayer(layerSizes);
    }
    nn.Initialize();
  }
  if (begin.contains(mouseX, mouseY)) {
    nn.training = !nn.training;
    if (nn.training)
      thread("threadedTraining");
  }
}
void mouseDragged() {
  speed.mouseDragged();
}
void mouseReleased() {
  speed.mouseReleased();
}
void keyPressed() {
  numInputs.keyPressed();
  numOutputs.keyPressed();
  numLayers.keyPressed();
  nodesPerLayer.keyPressed();
  for (int i = 0; i < inputs.size(); i++) {
    inputs.get(i).keyPressed();
  }
}

void threadedTraining() {
  while (nn.training) {
    if (iteration % 16 == 0) { 
      biggestErrorForEpoch = 0;
    }
    double[] input = project1InputSeeded(iteration);
    double[] output = project1Output(input);
    if (nn.IsInitialized()) {
      nn.Train(new double[][]{input}, 
        new double[][] {output});
    }
    if (nn.currentCost > biggestErrorForEpoch){
      biggestErrorForEpoch = nn.currentCost;
    }
    
    if (iteration % 16 == 15) { 
      if (biggestErrorForEpoch <= 0.05) {
        nn.training = false;
        return;
      }
    }
    
    iteration++;
    delay(speed.getValue());
  }
}

double[] randomInputHomework1Question1() {
  //bias, x, y, z
  return new double[] {1, (int)pow(-1, (int)random(2)), (int)pow(-1, (int)random(2)), (int)pow(-1, (int)random(2))};
}
double[] outputHomework1Question1(double[] in) {
  if (in[1] == -1 && in[2] == 1) {
    return new double[]{in[3]};
  } else return new double[]{-1};
}
double[] randomInputHomework1Question2() {
  //bias, u, v, w, x where uv and wx are binary numbers
  return new double[] {1, (int)random(2), (int)random(2), (int)random(2), (int)random(2)};
}
double[] outputHomework1Question2(double[] in) {
  //yz (as binary number)
  return new double[] {(in[2] == 1 ^ in[4] == 1) ^ in[1] == 1 ^ in [3] == 1? 1 : 0, 
    in[4] == 1 ^ in[2] == 1  ? 1 : 0  };
}
double[] project1Input() {
  // b, x1, x2, x3, x4, x5, x6, x7, x8
  return new double[] {(int)random(2), 
    (int)random(2), (int)random(2), 
    (int)random(2)};
}
double[] project1InputSeeded(int i) {
  // b, x1, x2, x3, x4, x5, x6, x7, x8
  int q = (i % 16) / 8;
  int w = (i % 8) / 4;
  int r = (i % 4) / 2;
  return new double[] {q, w, r, i % 2};
}
double[] project1Output(double[] input) {
  int num1s = 0;
  for (int i = 0; i < input.length; i++) {
    if (input[i] == 1) num1s++;
  }
  return new double[] {num1s % 2 == 1 ? 1 : 0};
}