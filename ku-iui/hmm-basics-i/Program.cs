﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace COMP537_Assignment_3____HMM {
  class Program {

    [STAThread] // Enables use of System.Windows.Forms dialogs
    static void Main(string[] args) {

      // Init vars for the HMM and observation sequence
      HiddenMarkovModel model = null;
      int[] obs = null;
   
      // Init var to read user input
      string command;

      // Title and help text
      string doctext = @"
COMP 537 Assignment 3: HMM Basics
=================================
by M.A. Baytaş

Commands
--------
                          
loadmodel        loads model from text file
loadobs          loads observation sequence from text file
showmodel        prints current model parameters
showobs          prints current observation sequence
obsv_prob        prints probability of observation sequence given model
obsv_prob log    prints probability of observation sequence given model
viterbi          prints best state sequence, number of state transitions
                 and likelihood, given observation sequence and model
quit             quits
help             shows available commands";

      Console.WriteLine(doctext);

      while (true) {
        Console.WriteLine();
        Console.Write("> ");
        command = Console.ReadLine();

        switch (command) {
        #region CLI Switch

          case "help":
            Console.WriteLine();
            Console.WriteLine(doctext);
            break;

          case "quit":
            Console.WriteLine();
            Console.WriteLine("Bye!");
            Thread.Sleep(1500);
            Environment.Exit(0);
            break;

          case "loadmodel":
            Console.WriteLine();

            using (OpenFileDialog fd = new OpenFileDialog()) {
              fd.Filter = "Plain text files (*.txt)|*.txt|All files (*.*)|*.*";
              if (fd.ShowDialog() == DialogResult.OK) {
                try {
                  List<String> lines = new List<string>();
                  using (StreamReader sr = new StreamReader(fd.FileName)) {
                    string line = sr.ReadLine();
                    while (line != null) {
                      lines.Add(line);
                      line = sr.ReadLine();
                    };
                  }
                  HiddenMarkovModel temp = new HiddenMarkovModel();
                  temp.LinesToModel(lines);
                  model = temp;
                  Console.WriteLine("Model loaded from:\n" + fd.FileName);
                }
                catch (Exception) {
                  Console.WriteLine("ERROR: Cannot read from file!");
                }
              }
              else Console.WriteLine("Aborted.");
            }
            break;

          case "loadobs":
            Console.WriteLine();

            using (OpenFileDialog fd = new OpenFileDialog()) {
              fd.Filter = "Plain text files (*.txt)|*.txt|All files (*.*)|*.*";
              if (fd.ShowDialog() == DialogResult.OK) {
                using (StreamReader sr = new StreamReader(fd.FileName)) {
                  try {
                    string line = sr.ReadLine();
                    obs = new int[line.Split().Length];
                    if (sr.ReadLine() != null) throw (new Exception());
                    for (int i = 0; i < line.Split().Length; i++) {
                      obs[i] = int.Parse(line.Split()[i]);
                    }
                    Console.WriteLine("Observation sequence loaded from:\n" + fd.FileName);
                  }
                  catch (Exception) {
                    Console.WriteLine("ERROR: Cannot read from file!");
                  }
                }
              }
              else Console.WriteLine("Aborted.");
            }
            break;

          case "showmodel":
            Console.WriteLine();

            if (model != null) {
              Console.WriteLine("Currently loaded model parameters:");
              Console.WriteLine();

              // Print A
              Console.WriteLine("  A:");
              for (int i = 0; i < model.A.GetLength(0); i++) {
                Console.Write("    ");
                for (int j = 0; j < model.A.GetLength(1); j++) {
                  Console.Write(model.A[i, j] + "  ");
                }
                Console.WriteLine();
              }
              Console.WriteLine();
              // Print B
              Console.WriteLine("  B:");
              for (int i = 0; i < model.B.GetLength(0); i++) {
                Console.Write("    ");
                for (int j = 0; j < model.B.GetLength(1); j++) {
                  Console.Write(model.B[i, j] + "  ");
                }
                Console.WriteLine();
              }
              Console.WriteLine();
              // Print pi
              Console.WriteLine("  pi:");
              Console.Write("    ");
              for (int i = 0; i < model.Pi.Length; i++) {
                Console.Write(model.Pi[i] + "  ");
              }
              Console.WriteLine();
            }
            else {
              Console.WriteLine("No model loaded!");
            }
            break;

          case "showobs":
            Console.WriteLine();

            if (obs != null) {
              Console.WriteLine("Currently loaded observation sequence:");
              Console.WriteLine();

              Console.Write("  ");
              for (int i = 0; i < obs.Length; i++) Console.Write(obs[i] + "  ");
              Console.WriteLine();
            }
            else {
              Console.WriteLine("No observation sequence loaded!");
            }

            break;

          case "obsv_prob":
            Console.WriteLine();
            Console.WriteLine("Likelihood of the currently loaded observation sequence,\ngiven the currently loaded model:");
            Console.WriteLine();
            Console.Write("  ");
            double loglike = Forward(model, obs, false);
            Console.WriteLine(loglike);
            break;

          case "obsv_prob log":
            Console.WriteLine();
            Console.WriteLine("Log likelihood of the currently loaded observation sequence,\ngiven the currently loaded model:");
            Console.WriteLine();
            Console.Write("  ");
            double like = Forward(model, obs, true);
            Console.WriteLine(like);
            break;

          case "viterbi":
            Console.WriteLine();
            // Calculate and print best state sequence
            // Also init prob variable to store probability
            Console.WriteLine("Best state sequence for currently loaded observation sequence,\ngiven the currently loaded model:");
            Console.WriteLine();
            Console.Write("  ");
            double prob = 0;
            int[] result = Viterbi(model, obs, out prob);
            for (int i = 0; i < result.Length; i++) Console.Write(result[i] + "  ");
            Console.WriteLine();
            Console.WriteLine();
            // Calculate and print number of state transitions
            Console.WriteLine("Number of state transitions:");
            int trans = 0;
            for (int i = 1; i < result.Length; i++) if (result[i] != result[i - 1]) trans++;
            Console.WriteLine();
            Console.Write("  ");
            Console.WriteLine(trans);
            Console.WriteLine();
            Console.WriteLine();
            // Print probability
            Console.WriteLine("Probability of the currently loaded observation sequence\nbeing generated by this state sequence:");
            Console.WriteLine();
            Console.Write("  ");
            Console.WriteLine(prob);
            Console.WriteLine();
            break;

          default:
            Console.WriteLine("Does not compute.");
            Console.WriteLine("Type 'help' for list of available commands.");
            break;
        } 
        #endregion

      }

    }

    /// <summary>
    ///   Returns the probability that a given HMM has generated the given sequence, using forward procedure.
    /// </summary>
    /// <param name="model" />A HiddenMarkovModel object
    /// <param name="obs" />Observation sequence as an integer array
    /// <param name="log" />Toggles return of log / plain likelihoods
    /// <returns>The probability (double) that the given HMM has generated the given sequence</returns>
    static double Forward(HiddenMarkovModel model, int[] obs, bool log) {

      double[,] A = model.A;
      double[,] B = model.B;
      double[] pi = model.Pi;
      int N = A.GetLength(0);
      int T = obs.Length;   
      double prob = 0;

      // 1) Initialization
      double[,] alpha = new double[T, N];
      for (int i = 0; i < N; i++) alpha[0, i] = pi[i] * B[i, obs[0]];
      
      // 2) Induction
      for (int t = 0; t < T - 1; t++) {
        for (int j = 0; j < N; j++) {
          double sum = 0;
          for (int i = 0; i < N; i++) sum += alpha[t, i] * A[i, j];
          alpha[t + 1, j] = sum * B[j, obs[t + 1]];
        }
      }

      // Scaling
      // Implementation of scaling coefficients
      // slightly differentthan Rabiner tutorial
      double[] c = new double[T];
      for (int t = 0; t < T; t++) {
        for (int i = 0; i < N; i++) {
          c[t] += alpha[t, i];
        }
      }

      // 3) Termination
      if (log) for (int t = 0; t < T; t++) prob += Math.Log(c[t]);
      else for (int i = 0; i < N; i++) prob += alpha[T - 1, i];

      // Return result
      return prob;
    }

    /// <summary>
    ///   Calculates the most likely sequence of hidden states that produced the given observation sequence, using Viterbi algorithm.
    /// </summary>
    /// <param name="model" />A HiddenMarkovModel object
    /// <param name="obs" />Observation sequence as an integer array
    /// <returns>The state sequence (integer array) that is most likely to produce the observation sequence.</returns>
    static int[] Viterbi(HiddenMarkovModel model, int[] obs, out double prob) {

      double[,] A = model.A;
      double[,] B = model.B;
      double[] pi = model.Pi;
      int T = obs.Length;
      int N = A.GetLength(0);
      double[,] delta = new double[T, N];
      int[,] psi = new int[T, N];
      int[] q = new int[T];

      // 1) Initialization
      for (int i = 0; i < N; i++) {
        delta[0, i] = pi[i] * B[i, obs[0]];
        psi[0, i] = 0;
      }

      // 2) Recursion
      for (int t = 1; t < T; t++) {
        for (int j = 0; j < N; j++) {
          
          double max = 0;
          double alt = 0;
          int argmax = 0;
          for (int i = 0; i < N; i++) {
            alt = delta[t - 1, i] * A[i, j];
            if (alt > max) {
              max = alt;
              argmax = i;
            }
          }

          delta[t, j] = max * B[j, obs[t]];
          psi[t, j] = argmax;
          
        }    
      }

      // 3) Termination
      double max1 = 0;
      double alt1 = 0;
      int argmax1 = 0;
      for (int i = 0; i < N; i++) {
        alt1 = delta[T - 1, i];
        if (alt1 > max1) {
          max1 = alt1;
          argmax1 = i;
        }
      }
      q[T - 1] = argmax1;
      prob = max1;

      // 4) Backtracking
      for (int t = T - 2; t >= 0; t--) q[t] = psi[t + 1, q[t + 1]];

      // Return result
      return q;

    }

  }

  /// <summary>
  ///   Hidden Markov Model class
  /// </summary>
  class HiddenMarkovModel {
    public double[,] A;
    public double[,] B;
    public double[] Pi;

    /// <summary>
    ///   Sets model parameters from alist of strings read from the model file
    /// </summary>
    public void LinesToModel(List<String> lines) {

      try {

        // Find out where data begins
        int A_index = lines.IndexOf("A");
        int B_index = lines.IndexOf("B");
        int pi_index = lines.IndexOf("Pi");

        // Get lines with data
        List<String> A_list = lines.GetRange(A_index + 2, B_index - A_index - 3);
        List<String> B_list = lines.GetRange(B_index + 2, pi_index - B_index - 3);
        String pi_string = lines[pi_index + 2];
        
        // Parse A
        A = new double[A_list.Count, A_list[0].Split().Length];
        for (int i = 0; i < A_list.Count; i++) {
          for (int j = 0; j < A_list[0].Split().Length; j++) {
            A[i, j] = double.Parse(A_list[i].Split()[j]);
          }
        }
        
        // Parse B
        B = new double[B_list.Count, B_list[0].Split().Length];
        for (int i = 0; i < B_list.Count; i++) {
          for (int j = 0; j < B_list[0].Split().Length; j++) {
            B[i, j] = double.Parse(B_list[i].Split()[j]);
          }
        }

        // Parse Pi
        Pi = new double[pi_string.Split().Length];
        for (int i = 0; i < pi_string.Split().Length; i++) {
          Pi[i] = double.Parse(pi_string.Split()[i]);
        }

      }
      catch (Exception) {
        throw;
      }

    }

  }

}