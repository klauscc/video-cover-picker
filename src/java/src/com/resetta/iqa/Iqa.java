package com.resetta.iqa;

public class Iqa {

  public native String detectBestImageIndir(String imageDir);


  static {
    try {
      System.loadLibrary("iqa_jni");
    }
    catch (UnsatisfiedLinkError error) {
      // Output expected UnsatisfiedLinkErrors.
      System.out.println(error);
    } catch (Error | Exception error) {
      // Output unexpected Errors and Exceptions.
      System.out.println(error);
    }
  }
  public static void main(String[] args) {
    Iqa iqa = new Iqa();
    String videoPath = "../dataset/video1000/samples/7704878478691672952/";
    System.out.println(iqa.detectBestImageIndir(videoPath));
  }
}
