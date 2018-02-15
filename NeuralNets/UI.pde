

class TextInput {
  public boolean selected = false;
  public String text = "0"; 
  public Rectangle r;
  public TextInput(int x, int y, int width, int height) {
    r = new Rectangle(x, y, width, height);
    r.c = color(255);
  }
  public void draw() {
    noStroke();
    r.draw(); 
    fill(0);
    textSize(r.height - 5);
    text(text, r.x + 1, r.y + r.height);
    if (selected) {
      noFill();
      stroke(0, 255, 0);
      rect(r.x, r.y, r.width, r.height);
    } else {
      noFill();
      stroke(0);
      rect(r.x, r.y, r.width, r.height);
    }
  }
  public void mousePressed() {
    if (r.contains(mouseX, mouseY)) {
      selected = true;
    } else selected = false;
  }
  void keyPressed() {
    if (selected) {
      if (keyCode == BACKSPACE) {
        if (text.length() > 0) {
          text = text.substring(0, text.length()-1);
        }
      } else if (keyCode == DELETE) {
        text = "";
      } else if (keyCode != SHIFT
        && textWidth(text + key) < r.width) {
        text = text + key;
      }
    }
  }
}

class IntSlider {
  int min, max, x, y, size;
  String name;
  Rectangle background;
  Rectangle slider; 
  boolean holding = false;
  int sliderSize;
  public IntSlider(String name, int min, int max, int x, int y, int size) {
    this.min = min;
    this.max = max;
    this.name = name;
    this.x = x;
    this.y = y; 
    this.size = size;
    sliderSize = size / 15;
    background = new Rectangle(x, y, size, sliderSize);
    slider = new Rectangle(x, y - 5, sliderSize / 2, sliderSize + 10);
    slider.c = color(0);
    background.c = color(0);
  }
  public void setMin(int n) {
    min = n;
  }
  public void setMax(int n) {
    max = n;
  }
  public int getValue() {
    int closest = 0;
    float dist = distance(x, y, slider.x, slider.y);
    for (int i = 1; i <= max - min; i += 1) {
      float d = distance(slider.x, slider.y, x + i * (size / (float)(max - min)), y);
      if (dist > d) {
        dist =  d;
        closest = i;
      }
    }
    return min + closest;
  }
  public void draw() {

    background.draw();
    slider.draw();
    fill(255);
    textSize(size / 20);
    text(min, x - sliderSize * (min + "").length() - 5, y + sliderSize);
    text(max, x + size + 5, y + sliderSize);
    text(getValue(), slider.x, slider.y + slider.height + sliderSize);
  }

  public void mousePressed() {
    if (slider.contains(mouseX, mouseY)) {
      holding = true;
    }
  }
  public void mouseDragged() {
    if (holding) {
      slider.x = mouseX;
      if (slider.x < x) slider.x = x;
      if (slider.x > x + size) slider.x = x + size;
    }
  }
  public void mouseReleased() {
    holding = false;
  }
}

class Point {
  public int x, y;
  public Circle circle;
  public color c;
  public Point(int x, int y) {
    this.x = x; 
    this.y = y;
    circle = new Circle(x, y, 3, 3);
    circle.c = color(0);
  }
  public void draw() {
    circle.draw();
  }
}
class MoveablePoint extends Point {
  public Circle circle;
  public boolean holding = false;
  public Rectangle bounds;
  public MoveablePoint(int x, int y, Rectangle bounds) {
    super(x, y);
    this.x = x;
    this.y = y;
    circle = new Circle(x, y, 12, 12);
    circle.c = color(0, 0, 255);
    this.bounds = bounds;
  }
  public Point toPoint() {
    return new Point(x, y);
  }
  public void draw() {
    circle.c = color(0, 0, 255);
    if (holding) circle.c = color(255, 0, 0);
    circle.draw();
  }
  public void mousePressed() {
    if (circle.contains(mouseX, mouseY)) {
      holding = true;
    }
  }
  public void mouseDragged() {
    if (holding) {
      x = mouseX;
      y = mouseY;
      circle.x = mouseX;
      circle.y = mouseY;
      if (circle.x < bounds.x) {
        circle.x = bounds.x;
        x = bounds.x;
      }
      if (circle.y < bounds.y) {
        circle.y = bounds.y;
        y = bounds.y;
      }
      if (circle.x > bounds.x + bounds.width) {
        circle.x = bounds.x + bounds.width;
        x = bounds.x + bounds.width;
      }
      if (circle.y > bounds.y + bounds.height) {
        circle.y = bounds.y + bounds.height;
        y = bounds.y + bounds.height;
      }
    }
  }
  public void mouseReleased() {
    holding = false;
  }
}

class Rectangle {
  public int x, y, width, height;
  public color c;
  public Rectangle(int x, int y, int width, int height) {
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
  }
  public void draw() {
    fill(c);
    rect(x, y, width, height);
    noFill();
    stroke(0);
    rect(this.x, this.y, this.width, this.height);
  }
  boolean contains(float x, float y) {
    return x >= this.x && x <= this.x + width && y >= this.y && y <= this.y + height;
  }
}

class Circle {
  public int x, y, width, height;
  public color c;
  public Circle(int x, int y, int width, int height) {
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
  }
  boolean contains(float x, float y) {
    return x >= this.x && x <= this.x + width && y >= this.y && y <= this.y + height;
  }
  public void draw() {
    fill(c);
    ellipse(x, y, width, height);
    noFill();
    stroke(0);
    ellipse(this.x, this.y, this.width, this.height);
  }
}