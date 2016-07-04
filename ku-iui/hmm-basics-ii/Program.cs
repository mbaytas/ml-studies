using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace HMM_Learning {
  class Program {

    // Enables use of System.Windows.Forms dialogs
    [STAThread]

    static void Main(string[] args) {

      // Init vars for observation sequences and model
      List<int[]> obss = new List<int[]>();
      Dictionary<string, object> model = new Dictionary<string,object>();

      #region CLI

      // Title and help text
      string doctext = @"
COMP 537 Assignment 4: HMM Learning
=================================
by M.A. Baytaş

Commands
--------
                          
loadobs          loads observation sequences from text file
showobs          prints current observation sequences
learn            trains a HMM using the given observation sequences
showmodel        prints the current HMM parameters
savemodel        saves the current HMM to a file
quit             quits
help             shows available commands";

      Console.WriteLine(doctext);

      while (true) {
        Console.WriteLine();
        Console.Write("> ");
        string command = Console.ReadLine();

        switch (command) {

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

          case "loadobs":
            Console.WriteLine();

            using (OpenFileDialog fd = new OpenFileDialog()) {
              fd.Filter = "Plain text files (*.txt)|*.txt|All files (*.*)|*.*";
              if (fd.ShowDialog() == DialogResult.OK) {
                try {
                  using (StreamReader sr = new StreamReader(fd.FileName)) {
                    string line;
                    while ((line = sr.ReadLine()) != null) {
                      obss.Add(Array.ConvertAll(line.Trim().Split(), int.Parse));
                    }
                  }
                  Console.WriteLine(obss.Count + " observation sequences loaded from:\n" + fd.FileName);
                }
                catch (Exception) {
                  Console.WriteLine("ERROR: Cannot read from file!");
                }
              }
              else Console.WriteLine("Aborted.");
            }
            break;

          case "showobs":
            Console.WriteLine();

            if (obss.Count != 0) {
              Console.WriteLine("Currently loaded observation sequences:");
              Console.WriteLine();
              foreach (int[] obs in obss) {
                Console.Write("{ ");
                for (int i = 0; i < obs.Length; i++) Console.Write(obs[i] + " ");
                Console.Write(" }");
                Console.WriteLine();
              }
              
            }
            else {
              Console.WriteLine("No observation sequence loaded!");
            }

            break;

          case "learn":
            Console.WriteLine();
            if (obss.Count != 0) {
              Console.WriteLine("Learning HMM from " + obss.Count + " observation sequences...");
              model = Learn(obss);
              Console.WriteLine("Done!");
              Console.WriteLine("Use showmodel to view and savemodel to save the HMM.");
              Console.WriteLine();
              break;
            }
            Console.WriteLine("Currently loaded model parameters:");

            break;

          case "showmodel":
            Console.WriteLine();

            if (model.Count != 0) {
              Console.WriteLine("Currently loaded model parameters:");
              Console.WriteLine();

              // Print A
              Console.WriteLine("  A:");
              for (int i = 0; i < ((double[,])model["A"]).GetLength(0); i++) {
                Console.Write("    ");
                for (int j = 0; j < ((double[,])model["A"]).GetLength(1); j++) {
                  Console.Write(((double[,])model["A"])[i, j] + "  ");
                }
                Console.WriteLine();
              }
              Console.WriteLine();
              // Print B
              Console.WriteLine("  B:");
              for (int i = 0; i < ((double[,])model["B"]).GetLength(0); i++) {
                Console.Write("    ");
                for (int j = 0; j < ((double[,])model["B"]).GetLength(1); j++) {
                  Console.Write(((double[,])model["B"])[i, j] + "  ");
                }
                Console.WriteLine();
              }
              Console.WriteLine();
              // Print pi
              Console.WriteLine("  pi:");
              Console.Write("    ");
              for (int i = 0; i < ((double[])model["Pi"]).Length; i++) {
                Console.Write(((double[])model["Pi"])[i] + "  ");
              }
              Console.WriteLine();
            }
            else Console.WriteLine("No model learned!");

            break;

          case "savemodel":

            if (model.Count != 0) {
              using (SaveFileDialog fd = new SaveFileDialog()) {
                fd.Filter = "Plain text files (*.txt)|*.txt|All files (*.*)|*.*";
                fd.FileName = "HMMdescription.txt";
                if (fd.ShowDialog() == DialogResult.OK) {
                  try {
                    using (StreamWriter sw = new StreamWriter(fd.FileName)) {
                      // Write A
                      sw.WriteLine("A");
                      sw.WriteLine();
                      for (int i = 0; i < ((double[,])model["A"]).GetLength(0); i++) {
                        for (int j = 0; j < ((double[,])model["A"]).GetLength(1); j++) {
                          sw.Write(((double[,])model["A"])[i, j] + " ");
                        }
                        sw.WriteLine();
                      }
                      sw.WriteLine();
                      // Write B
                      sw.WriteLine("B");
                      sw.WriteLine();
                      for (int i = 0; i < ((double[,])model["B"]).GetLength(0); i++) {
                        for (int j = 0; j < ((double[,])model["B"]).GetLength(1); j++) {
                          sw.Write(((double[,])model["B"])[i, j] + " ");
                        }
                        sw.WriteLine();
                      }
                      sw.WriteLine();
                      // Write Pi
                      sw.WriteLine("Pi");
                      sw.WriteLine();
                      for (int i = 0; i < ((double[])model["Pi"]).Length; i++) {
                        sw.Write(((double[])model["Pi"])[i] + " ");
                      }
                    }
                    Console.WriteLine("HMM written to file:\n" + fd.FileName);
                  }
                  catch (Exception) {
                    Console.WriteLine("ERROR: Cannot write to file!");
                  }
                }
                else Console.WriteLine("Aborted.");
              }
            }
            else Console.WriteLine("No model learned!");
            
            break;

          default:
            Console.WriteLine("Does not compute.");
            Console.WriteLine("Type 'help' for list of available commands.");
            break;

        }
      }
      #endregion

    }

    /// <summary>
    ///    Learns and returns a HMM (as a dictionary) from a set of observation sequences
    /// </summary>
    /// <param name="obss" /> The observation sequences as a list of integer arrays
    /// <returns>Hidden Markov model as Dictionary<string, object></returns>
    static Dictionary<string, object> Learn(List<int[]> obss) {

      // Store number of observation sequences
      int L = obss.Count;

      // Estimate N assuming no of states = no of unique observations
      int N = obss.SelectMany(a => a).Distinct().Count();

      // Estimate Pi
      double[] pi = new double[N];
      for (int i = 0; i < N; i++) {
        foreach (int[] obs in obss) if (obs[0] == i) pi[i]++;
        pi[i] /= L;
      }

      // Init Random for initial estimates of A and B
      Random r = new Random();
      // Initial estimate for A
      double[,] A = new double[N, N];
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          A[i, j] = r.Next();
        }
      }
      for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) sum += A[i, j];
        for (int j = 0; j < N; j++) A[i, j] /= sum;
      }
      // Initial estimate for B
      double[,] B = new double[N, N];
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          B[i, j] = r.Next();
        }
      }
      for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) sum += B[i, j];
        for (int j = 0; j < N; j++) B[i, j] /= sum;
      }

      // Loop Baum-Welch for 10,000 times
      // I cap the number of iterations instead of maximizing probabilities
      // Calculating probabilities causes underflow without scaling and overflow with scaling
      for (int iter = 0; iter < 10000; iter++) {

        Console.CursorLeft = 0;
        Console.Write((iter + 1) + " / " + 10000 );

        // Init Gamma and Xi vars
        List<double[,]> gammas = new List<double[,]>();
        List<double[, ,]> xis = new List<double[, ,]>();

        // Produce a Gamma and Xi matrix for every observation sequence
        foreach (int[] obs in obss) {

          int T = obs.Length;

          // Calculate forward and backward vars and probability of the sequence
          double[,] alpha = Forward(A, B, pi, obs);
          double[,] beta = Backward(A, B, pi, obs);
          double prob = 0.0;
          for (int i = 0; i < N; i++) prob += alpha[T - 1, i];

          // Gamma variable
          double[,] gamma = new double[T, N];
          for (int t = 0; t < T; t++) {
            for (int i = 0; i < N; i++) {
              gamma[t, i] = alpha[t, i] * beta[t, i] / prob;
            }
          }
          gammas.Add(gamma);

          // Xi variable
          double[, ,] xi = new double[T, N, N];
          for (int t = 0; t < T - 1; t++) {
            for (int i = 0; i < N; i++) {
              for (int j = 0; j < N; j++) {
                xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, obs[t + 1]] * beta[t + 1, j] / prob;
              }
            }
          }
          xis.Add(xi);

        }

        // Adjust A
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++) {
            double num = 0.0;
            double den = 0.0;
            for (int l = 0; l < L; l++) {
              for (int t = 0; t < 9; t++) {
                num += xis[l][t, i, j];
                den += gammas[l][t, i];
              }
            }
            A[i, j] = num / den;
          }
        }
        // Adjust B
        for (int j = 0; j < 2; j++) {
          for (int k = 0; k < 2; k++) {
            double num = 0.0;
            double den = 0.0;
            for (int l = 0; l < L; l++) {
              for (int t = 0; t < obss[0].Length; t++) {
                if (obss[l][t] == k) num += gammas[l][t, j];
                den += gammas[l][t, j];
              }
            }
            B[j, k] = num / den;
          }
        }

      }

      return new Dictionary<string, object>() {
        {"A", A},
        {"B", B},
        {"Pi", pi}
      };

    }

    /// <summary>
    ///   Returns the backward probabilities matrix for a given HMM and a given observation sequence
    /// </summary>
    /// <param name="A" /> HMM transition probabilities matrix, calculated in Main()
    /// <param name="B" /> HMM emission probabilities matrix, calculated in Main()
    /// <param name="pi" /> HMM initial state probabilities vector, calculated in Main()
    /// <param name="obs" />Observation sequence as an integer array
    /// <returns>Forward probabilities matrix as double[,]</returns>
    static double[,] Forward(double[,] A, double[,] B, double[] pi, int[] obs) {

      int N = A.GetLength(0);
      int T = obs.Length;

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

      return alpha;
    }

    /// <summary>
    ///    Returns the backward probabilities matrix for a given HMM and a given observation sequence
    /// </summary>
    /// <param name="A" /> HMM transition probabilities matrix, calculated in Main()
    /// <param name="B" /> HMM emission probabilities matrix, calculated in Main()
    /// <param name="pi" /> HMM initial state probabilities vector, calculated in Main()
    /// <param name="obs" />Observation sequence as an integer array
    /// <returns>Backward probabilities matrix as double[,]</returns>
    static double[,] Backward(double[,] A, double[,] B, double[] pi, int[] obs) {

      int N = A.GetLength(0);
      int T = obs.Length;

      // 1) Initialization
      double[,] beta = new double[T, N];
      for (int i = 0; i < N; i++) beta[T - 1, i] = 1;
      // 2) Induction
      for (int t = T - 2; t >= 0; t--) {
        for (int i = 0; i < N; i++) {
          beta[t, i] = 0;
          for (int j = 0; j < N; j++) beta[t, i] += A[i, j] * B[j, obs[t + 1]] * beta[t + 1, j];
        }
      }

      return beta;
    }

  }
}
