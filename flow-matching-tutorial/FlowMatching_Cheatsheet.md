# Flow Matching 


## Ordinary Differential Equation (ODE)

An ODE describes how something changes over time (or another continuous variable). The key idea is that an ODE describes the rate of change of a quantity and from that it is possible to calculate how the quantity evolves. 

$$
\frac{dx}{dt} = v_\theta (x,t)
$$

At position $x$ and time $t$, the networks specifies in which direction and how quickly something should move. The learned function can be visualized as velocity field, where every point $(x, t) has an associated velocity. Following the velocities results in a path which is the solution of the ODE. 


Flow matching does not solve ODEs analytically, instead we take small numerical steps. ODEs can be seen as mathematical rule describing how to continuously move x from the starting distribution toward the target distribution. 


## Flow Matching Design Choices

```text
SOURCE DISTRIBUTION
      │
      │  $x_0 ~ p_0$
      ▼
┌──────────────────────┐
│  1. Coupling         │  How do $x_0$ and $x_1$ correspond?
└──────────────────────┘
      │
      ▼
┌──────────────────────┐
│  2. Probability path │  How do we get from $x_0 -> x_1$ ?
└──────────────────────┘
      │
      ▼
┌──────────────────────┐
│  3. Target vector    │  What velocity should the model learn?
│     field            │
└──────────────────────┘
      │
      ▼
┌──────────────────────┐
│  4. Neural network   │  $v_\theta (x,t)$
└──────────────────────┘
      │
      ▼
┌──────────────────────┐
│  5. Training         │  How is $v \theta$ optimized?
└──────────────────────┘
      │
      ▼
┌──────────────────────┐
│  6. ODE solver       │  How do we generate samples?
└──────────────────────┘
      │
      ▼
   $x_1 ~ pdata$
```


#### Historical foundation


1. Normalizing flows 

Early flow-based generative models tried to learn an invertible transformation $z -> x$ from a simple distribution to the data distribution. The transformation is usually constructed from explicit invertible layers. The problem is that designing and evaluating these transformations can become restrictive. 

2. Continuous Normalizing Flows 

The next important idea was to define the transformation  through an ODE (see above). This is the mathematical foundation underneath modern Flow Matching. Instead of saying:

```text
z -> layer 1 -> layer 2 -> layer 3 -> x

we define: 

z -------------------------> x
       continuous trajectory
```


3. Problem with directly training CNFs

A natural approach would be: 

1. Start with noise. 
2. Integrate the ODE. 
3. Compare the generated result with data. 
4. Backpropagate through the ODE solver. 

Since this is computationally expensive, Flow Matching introduces a clever alternative: Don't simulate the ODE during training. Construct a known path between samples and directly train the network to reproduce its velocity field (Central idea of 2022 Flow Matching paper). 


