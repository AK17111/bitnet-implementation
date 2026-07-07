# BitNet Understanding

Notes on things i need to understand 

-> a Transformer processes a sequence of tokens. Each token gets converted into a vector of numbers (called an embedding) — this vector is what actually flows through all those matrix multiplications (the nn.Linear / BitLinear layers) and attention calculations.

nn.Linear -> nn.Linear is the name of a specific building block in PyTorch (a popular deep learning software library). It represents exactly that matrix multiplication operation: input vector in, multiply by a weight matrix, output vector out (optionally add a bias). It's called "Linear" because a matrix multiplication is a linear transformation, mathematically speaking.

The paper says BitLinear is a drop in replacement

Why do you need BitLinear? - Normally, each number (weight) in a matrix is stored as a fairly precise number — e.g., FP16 or FP32, meaning 16 or 32 bits (binary digits) are used to represent each single number, allowing lots of possible values (positive, negative, decimals, huge range).
"1-bit" means each weight is squeezed down to just one bit — meaning it can only be one of two possible values. In this paper specifically, weights become either +1 or −1 (see Equation 1-2 in the paper: the Sign function). That's it — no shades of gray, just a binary switch for every single weight in the matrix.
This is extreme compression: instead of needing 16 or 32 bits to store each number, you need essentially 1. That directly shrinks:

Memory footprint — the model file/weights take far less space.
Energy/compute cost — multiplying by +1 or −1 is trivial (it's just a sign flip), so the expensive multiplication step is nearly eliminated; the paper notes this in Section 2.3, where BitNet's energy cost is dominated by cheap addition instead of costly multiplication.


What does "Training a model" mean in this context. 
"Training a model" means: automatically adjusting all those weight numbers (in the matrices we just talked about) so that the model gets better at some task — in this case, predicting the next token in a piece of text.
Remember, a nn.Linear layer starts with a weight matrix full of random numbers. A freshly initialized Transformer is basically useless — feed it "the cat sat on the" and it'll predict garbage. Training is the process that turns those random numbers into numbers that actually produce good predictions.
How does that work?
1. Give it an example and see what it predicts
Take a piece of real text, e.g. "the cat sat on the mat." Feed the model everything up to "the," and ask it to predict what token comes next.
2. Measure how wrong it was
The model outputs a guess (technically, a probability for every possible token in its vocabulary). We compare this to the actual next word ("mat") and calculate a number called the loss — a measure of how wrong the prediction was. Big error → big loss. Good prediction → small loss. (You'll see "Loss" plotted constantly in this paper — Figure 1, Figure 3 — that's this exact number, averaged over lots of examples.)
3. Figure out how to adjust the weights to reduce that error
This is the clever part, done via an algorithm called backpropagation combined with gradient descent. Without getting into the calculus: for every single weight number in every matrix in the entire network, the algorithm computes "if I nudge this specific number up a tiny bit, does the loss get better or worse, and by how much?" That nudge-direction-and-size is called a gradient.
4. Nudge every weight slightly in the direction that reduces the loss
Do this for millions/billions of weights simultaneously, a tiny step at a time.
5. Repeat, over and over, on tons of examples
Do steps 1-4 for a new chunk of text, then another, then another — often billions of times ("training updates" — the paper mentions "40K" updates, each update processing a big batch of tokens like 256K tokens per sample). Each repetition nudges the weights a little more toward "good at predicting text."
Over time, this process shapes the random noise in the weight matrices into structured patterns that actually encode grammar, facts, reasoning patterns, etc.

BitNet and the FP16 Transformers it's compared against are all trained as autoregressive language models, meaning their one job is: given all the tokens so far, predict the next one. The paper says this directly: "We train a series of autoregressive language models with BitNet of various scales."


LOSS FUNCTION, AND HOW IT IS CALCULATED -
The actual formula almost every language model uses is called cross-entropy loss, and it's:
loss = -log(probability assigned to the correct token)
Why not just 1 - probability? A few reasons this log version is better:

It punishes confident wrongness much more harshly. If the model assigns "mat" a probability of 0.5, -log(0.5) ≈ 0.69. If it assigns "mat" a probability of 0.01 (very confident it's something else), -log(0.01) ≈ 4.6 — a much bigger loss. With plain 1 - probability, going from 0.5 to 0.01 only changes the loss from 0.5 to 0.99 — barely a difference. The log version creates a much steeper penalty as the model gets more confidently wrong, which gives training a stronger signal to correct big mistakes.
It matches probability theory nicely — cross-entropy loss is mathematically tied to "how surprised should the model be by the true answer," which is exactly what you want to minimize.


REVISE WHAT IS A LINEAR TRANSFORMATION IN MATRIX MULTIPLICATION, and LOSS FUNCTION 
