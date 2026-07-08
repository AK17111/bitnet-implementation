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

Another reason why model quantization is a promising solution is that on distributed systems, the inter-device communication overhead can significantly impact the inference latency and energy consumption.

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


FFN stands for Feed-Forward Network — it's the second of the two main building blocks inside each Transformer layer (the first being attention, which we covered earlier). You can see it labeled right in Figure 2(b) of the paper, sitting right after "Multi-Head Attention" in each layer.
Here's what it does, step by step:

Attention's job (recap): let each token gather relevant context from other tokens in the sequence.
FFN's job: take each token's vector (now enriched with context from attention) and process it further, independently for each token — no interaction between tokens happens here, unlike attention.

Concretely, an FFN is usually just two nn.Linear layers stacked with a non-linear function in between:

You can see this exact structure in Figure 2(b) — the paper's diagram shows BitLinear → GELU → BitLinear as the FFN block. The "expand then compress" pattern gives the network extra capacity to do complex transformations on each token's information before passing it to the next layer.
Why does this matter for BitNet? Because the FFN is where a huge chunk of a Transformer's total parameters and compute live — those expanded matrices are big. This is exactly why replacing nn.Linear with BitLinear inside the FFN (as well as attention) is where most of BitNet's memory and energy savings come from.


LOSS FUNCTION, AND HOW IT IS CALCULATED -
The actual formula almost every language model uses is called cross-entropy loss, and it's:
loss = -log(probability assigned to the correct token)
Why not just 1 - probability? A few reasons this log version is better:

It punishes confident wrongness much more harshly. If the model assigns "mat" a probability of 0.5, -log(0.5) ≈ 0.69. If it assigns "mat" a probability of 0.01 (very confident it's something else), -log(0.01) ≈ 4.6 — a much bigger loss. With plain 1 - probability, going from 0.5 to 0.01 only changes the loss from 0.5 to 0.99 — barely a difference. The log version creates a much steeper penalty as the model gets more confidently wrong, which gives training a stronger signal to correct big mistakes.
It matches probability theory nicely — cross-entropy loss is mathematically tied to "how surprised should the model be by the true answer," which is exactly what you want to minimize.


REVISE WHAT IS A LINEAR TRANSFORMATION IN MATRIX MULTIPLICATION, and LOSS FUNCTION @


nn.Linear: input vector → multiply by weight matrix → output vector. This operation is called "linear" because of a specific mathematical property: if you stack multiple linear operations back-to-back with nothing in between, the whole stack collapses down to being equivalent to just one single linear operation. Two matrix multiplications in a row = mathematically the same as one (bigger) matrix multiplication.
This is a problem: if a Transformer were made of only nn.Linear layers, no matter how many you stacked, the whole giant network would be mathematically equivalent to one single matrix multiply. All that depth and size would be wasted — you couldn't learn complex, curved, "if this then that" style patterns that real language requires.
A non-linear function breaks that collapse. It's a small function applied to each number individually that "bends" the output in some way, so stacking layers actually adds real expressive power. Two common non-linear functions:

ReLU: output = max(0, x) — if the number is negative, turn it into 0; if positive, leave it alone. Dead simple, but it's non-linear because of that "kink" at zero.
GELU: a smoother, more gradual version of the same idea — instead of a hard cutoff at zero, it smoothly transitions negative values toward zero. It tends to work a bit better in practice for Transformers, which is why the paper uses it (Figure 2b shows GELU sitting between the two BitLinear layers in the FFN).

So: Linear layer = weighted sum (matrix multiply). Non-linear layer = a simple per-number "bending" function with no learned weights, inserted between linear layers specifically to stop them from collapsing into one and to let the network represent complex patterns.
Step 2: LayerNorm
LayerNorm (Layer Normalization) is a step that takes a vector of numbers and rescales it so the numbers have a controlled, consistent range — specifically, it adjusts the vector so its average is 0 and its spread (variance) is 1, using this formula from the paper (Eq. 12):
LN(x) = (x - mean(x)) / sqrt(variance(x) + ε)
Why is this needed? As data flows through many stacked layers, the scale of the numbers can drift — some layers might make values huge, others tiny, causing training to become unstable (exploding or vanishing values). LayerNorm resets things to a predictable scale before each major operation, which keeps training stable.
Why BitNet especially needs it: this is explicitly explained in Section 2.1. Full-precision Transformers naturally keep output variance around 1 thanks to careful weight initialization. But once you binarize the weights to ±1, that natural stability gets disrupted — the paper works out the math (Equations 8–10) showing the output variance changes. To fix this, they insert LayerNorm right before the activation quantization step, so the numbers are "reset" to a known, stable range immediately before being squeezed down to low precision. Without it, the quantization step would be working with wildly unpredictable input scales and training would fall apart.
Step 3: QKV — Query, Key, Value
This is the internal machinery of attention (the mechanism that lets tokens "look at" other tokens, which I described earlier). Here's the intuition:
Every token generates three different vectors from itself, via three separate nn.Linear (or here, BitLinear) layers:

Query (Q): "What am I looking for?" — represents what kind of information this token wants to gather from others.
Key (K): "What do I have to offer?" — represents what kind of information this token contains, that other tokens might want.
Value (V): "Here's my actual content" — the actual information to be passed along if another token decides this one is relevant.

The mechanism works like a search: each token's Query is compared against every other token's Key (via a dot-product/similarity calculation) to get a relevance score. Those scores get turned into weights (via softmax, so they sum to 1), and then used to compute a weighted average of everyone's Value vectors. The result: each token ends up with a new vector that's a blend of the most relevant information from the rest of the sequence.
That's exactly what you see in Figure 2(b)'s zoomed-in box: the input feeds into three separate BitLinear blocks labeled Q, K, V, then those get combined inside the "Attention" block.
Step 4: "h Heads" — Multi-Head Attention
Instead of doing that Q/K/V attention process just once per layer, Transformers split it into several smaller, parallel copies called heads — that's the h in "h Heads" in Figure 2(b).
Why? Doing attention once forces the model to capture all types of relationships (grammar, meaning, position, reference, etc.) through a single lens. Splitting into, say, 8 or 32 heads (see Table 5 — "# Heads" column, e.g. the 6.7B model has 32 heads) lets each head specialize: one head might learn to track subject-verb agreement, another might track "which pronoun refers to which noun," another might track nearby-word relationships. Each head does its own smaller Q/K/V attention independently, and then all the heads' outputs get concatenated back together into one vector before moving on.
This is why Figure 2(b) shows the arrow labeled "h Heads" wrapping around the attention block — it's saying "this whole Q/K/V attention process happens h times in parallel, each with its own separate learned weights."
Step 5: "L-Layer" — stacking it all
Now zoom back out. One full "layer" of a Transformer = Multi-Head Attention block + Feed-Forward Network block (with residual connections/LayerNorm around them, though the paper simplifies this in the diagram). Figure 2(b) shows this whole unit boxed together and labeled "L-Layer" — meaning this entire attention+FFN combo gets repeated L times, stacked on top of each other, where L is however many layers the model has (see Table 5 — e.g., the 6.7B model has 32 layers, the 30B model has 48).
Each stacked layer refines the token representations a bit further — early layers might capture basic syntax, later layers more abstract meaning — similar to how each layer in an image-recognition network captures increasingly complex visual features.
Now, putting Figure 2 together as a whole
Figure 2(a) — BitLinear's internal computation flow (left box):
Input → LayerNorm → Absmax Quantization → multiply by 1-bit Weights → Dequantization (using β, γ) → Output
This is literally Equation 11 drawn as a flowchart: normalize the input for stability, compress it to low-bit-precision, multiply by the binary (±1) weight matrix, then rescale the result back up to a usable range using the scaling factors β and γ from Equations 3 and 12.
Figure 2(b) — the whole BitNet Transformer (right side):
Input 
  → [Q/K/V via 3 BitLinear layers, split into h Heads] → Attention 
  → (this whole attention block, plus a BitLinear-based FFN with GELU in the middle)
  → repeated L times (the "L-Layer" stack)
  → Output
Every red "BitLinear" box in the diagram is literally the BitLinear flow from part (a) — the diagram is showing you that every place a normal Transformer would use nn.Linear (the Q projection, K projection, V projection, the FFN's two linear layers, and the final output projection) gets replaced by BitLinear instead, while everything else in the architecture (the overall attention mechanism, the multi-head splitting, the layer stacking, GELU) stays exactly the same as a standard Transformer.


In figure 2(b) there is a bitlinear layer after the attention block, this is why - 

Each of the h heads runs its own independent, smaller attention process, and produces its own output vector. If the model's full hidden size is, say, 4096, and there are 32 heads, each head might work with a 128-dimensional slice (4096 ÷ 32 = 128).
The problem: you have h separate pieces, not one unified vector
After all h heads finish, you don't have one clean output — you have h separate small vectors sitting side-by-side (this step is called concatenation — literally gluing them back together end-to-end). Concatenating 32 heads of 128 dimensions each gets you back to a 4096-dimensional vector, so the size is right again, but there's a problem: each head computed its piece in total isolation from the others. Head 1 doesn't know what Head 2 found. They were never combined or allowed to interact — they're just stapled together.
Why that's a problem, and what the output BitLinear fixes
If you just fed that stapled-together vector straight into the next layer, you'd be relying on all future computation to somehow untangle "the first 128 numbers came from a head that specialized in X, the next 128 from a head specializing in Y" — with no mixing between them. That's wasteful and limits what the model can express.
So, right after concatenation, there's a linear layer (nn.Linear in a normal Transformer, BitLinear here) whose entire job is to mix all the heads' information back together — every output number becomes a learned combination of numbers from every head, not just one. This lets the model blend and weigh what each head discovered into a single, unified representation before passing it on.
This is a completely standard part of the original Transformer design (often called the output projection, sometimes written W_O in papers) — it's not something BitNet invented; BitNet just applies its ±1 binarization trick to this layer too, same as every other linear layer in the network.

