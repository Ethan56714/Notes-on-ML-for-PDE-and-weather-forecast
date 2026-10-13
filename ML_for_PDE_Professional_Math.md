# Machine Learning for Partial Differential Equations

English text-only study notes. Equations are transcribed as editable text; diagrams and screenshots are replaced by prose descriptions or mathematical expressions.

Scope: introductory PDE concepts; PDE discovery and PDE-FIND; closure modelling and noise robustness; Koopman operators; reduced-order modelling (ROM), SVD and Galerkin projection; operator inversion; Green's functions and eigenfunction expansions; neural operators, DeepONet, PINN and PINO.

## 1. Introduction: What Is a PDE?

A partial differential equation (PDE) is a mathematical framework for describing how a quantity depending on several independent variables changes. In physical applications, these variables often include spatial coordinates and time. For example, a weather state can be represented as a field

$$
u = u(x,y,z,t).
$$

The state of the atmosphere evolves over a spatial domain as time passes. Consequently, a PDE model of weather typically involves spatial variables x, y and z, and time t. An ordinary differential equation (ODE) usually describes change with respect to one independent variable, whereas a PDE includes partial derivatives with respect to multiple independent variables.

A governing equation describes the physical relationships that determine the evolution of the system. A PDE problem generally also requires initial conditions (ICs) and boundary conditions (BCs). The governing equation alone is usually not enough to determine a unique solution.

$$
\frac{\partial u}{\partial t}+\mathcal{N}(u,u_x,u_{xx},\ldots;x,t;\mu)=f(x,t),
$$

Here u(x,t) is the unknown field, N represents the combination of physical effects and may be nonlinear, μ denotes parameters, and f is a forcing term. A representative initial condition is

$$
u(x,0)=u_0(x),
$$

while boundary conditions specify the behaviour of the solution at the spatial boundary. Their exact form depends on the physical problem.

Linear PDEs often admit systematic analytical techniques, although solvability depends on the operator, domain and conditions. Nonlinear PDEs are generally more difficult and may require numerical approximation. A dynamical system is a mathematical description of how a system's state changes over time; it can be continuous or discrete, and may be described by ODEs or PDEs.

## 2. Why Use Machine Learning for PDEs?

Many modern PDE systems are nonlinear, complex and high-dimensional. Traditional analytical methods may not yield a closed-form solution, while numerical simulation can be computationally expensive. Machine learning can provide a proxy model that approximates aspects of a physical model at lower computational cost.

Three broad directions are useful to distinguish:

- PDE discovery: infer governing equations from observed data when the equations, or parts of them, are unknown.

- Governing equations and coarse-grained closures: identify missing terms or model the influence of unresolved small scales on a coarse-resolution simulation.

- Learning solutions or solution operators: train a model to approximate the mapping from a PDE problem's inputs to its solution.

Learning an explicit equation can be more interpretable than using a black-box predictor: a discovered governing equation makes the proposed relationships visible. However, interpretability and generalisation are not guaranteed automatically; they depend on the data, candidate terms, assumptions and validation.

## 3. PDE Discovery and Symbolic Regression

### 3.1 Sparse identification of nonlinear dynamics (SINDy)

SINDy is a data-driven method for identifying a parsimonious dynamical model. Given observations of a state u, one estimates derivatives such as

$$
u_t,\;u_x,\;u_{xx},\;u_{xxx},\;u^2,\;u\,u_x,\;\ldots
$$

These candidate quantities are assembled into a library of possible terms. For example,

$$
\Theta(u)=[1,u,u_x,u_{xx},u^2,u\,u_x,u_{xx}u,u^3,\ldots].
$$

The time derivative is then approximated as a linear combination of the candidate terms:

$$
u_t\approx \Theta(u)\xi.
$$

The vector ξ contains the coefficients to be identified. The assumption is that the true governing equation can be represented using only a small subset of the candidate library, so most coefficients should be zero. This is the sparsity assumption.

The method relies on two important assumptions: the candidate library is rich enough to include the terms needed to represent the true equation, and the true equation has a sparse representation in that library. If either assumption fails, sparse regression may not recover the correct governing equation.

### 3.2 PDE-FIND

PDE-FIND extends sparse equation discovery to PDEs. A typical workflow is:

1. Collect spatial-temporal observations of the system, such as velocity components u and v and vorticity ω.

1. Estimate the derivatives needed to form candidate terms.

1. Construct a library matrix Θ, with each row corresponding to an observation point and each column corresponding to a candidate PDE term.

1. Use sparse regression to estimate the coefficient vector ξ.

1. Compare the identified equation with known physics, where possible, and validate it on data not used for fitting.

$$
u_t\approx \Theta(u,u_x,u_{xx},u^2,u\,u_x,\ldots)\xi.
$$

The matrix view is important: each row records the values of candidate terms at one sampled point, while each column corresponds to one candidate term. Sparse regression selects the terms that are needed to explain the observed derivative.

### 3.3 Sparse regression and the L0 penalty

A representative objective is

$$
\min_{\xi}\;\|\Theta\xi-u_t\|_2^2+\lambda\|\xi\|_0.
$$

The first term measures data-fitting error. The L0 pseudo-norm ||ξ||0 counts the number of nonzero coefficients and therefore penalises models with too many active terms. It is different from the L1 norm, which sums the absolute magnitudes of the coefficients. Direct optimisation with an L0 penalty is generally difficult; practical methods use specialised or approximate sparse-regression procedures.

A PDE-FIND example can be built from the Navier–Stokes equations: data are collected, a library of possible nonlinear and derivative terms is formed, and sparse regression selects active terms. This can test whether the method recovers a known equation, but recovery depends on data quality, derivative estimation, the candidate library and the identifiability of the terms.

## 4. Closure Modelling and Coarse-Graining

In multiscale systems such as atmospheric flows, simulating every scale may be prohibitively expensive. A coarse-resolution model resolves large scales but omits small-scale degrees of freedom. The unresolved scales can still affect the resolved dynamics, so their net influence must be represented by a closure term.

$$
u_t=F(u)+C.
$$

Here F(u) represents the resolved physics and C is the unknown closure term representing the effect of unresolved physics. If u_t and F(u) can be estimated, then

$$
C=u_t-F(u).
$$

One can construct a candidate library for C and fit a sparse model, for example

$$
C\approx \Theta(u)\xi,
$$

$$
\min_{\xi}\;\|C-\Theta\xi\|_2^2+\lambda\|\xi\|_0.
$$

A simple illustrative result might be C = 0.3 u_x + 0.02 u_xx, although the appropriate terms and coefficients depend on the system and data. In weather and fluid modelling, large-eddy simulation (LES) is an example of a framework in which closure modelling is important.

PDE discovery can therefore be used not only to rediscover a known equation, but also to infer missing closure terms. The inferred closure should be treated as a model to validate, not as a guaranteed universal physical law.

## 5. Noise Robustness in PDE Discovery

Derivative estimation from noisy observations is a major challenge. Pointwise finite differences can amplify noise, and errors in derivatives can then distort the candidate library and the identified equation.

### 5.1 Sparse Bayesian learning and relevance vector machines

A relevance vector machine (RVM) approach can place a separate prior on each candidate coefficient:

$$
w_i\sim\mathcal{N}(0,\alpha_i^{-1}).
$$

The parameter α_i is a precision, the inverse of the prior variance. A large α_i strongly shrinks the corresponding coefficient towards zero, reducing the influence of that candidate term. Bayesian inference combines the prior with the data to obtain a posterior over coefficients.

### 5.2 Weak-form PDE discovery

In a strong-form formulation, the PDE is enforced pointwise. For example,

$$
u_t+\mathcal{N}(u)=0.
$$

A weak-form formulation instead multiplies the equation by a suitable test function φ and integrates over a domain:

$$
\int_{\Omega}[u_t+\mathcal{N}(u)]\phi\,dx=0.
$$

Integration by parts can transfer derivatives from noisy data to the test function. For example,

$$
\int_{\Omega}u_x\phi\,dx=-\int_{\Omega}u\phi_x\,dx+[u\phi]_{\partial\Omega}.
$$

Under suitable regularity and boundary assumptions, this integral formulation can be less sensitive to noisy pointwise derivative estimates. Weak form does not mean that smoothness and boundary assumptions disappear entirely; rather, the required regularity can be weaker than in a classical pointwise formulation.

### 5.3 Ensemble methods

Another approach is to repeat sparse regression on different subsets or resamples of the data. Robust bagging and inclusion probabilities can help identify terms that are selected consistently across repeated fits. The notes identify PySINDy as an open-source software library incorporating methods for data-driven system identification.

### 5.4 Spatially varying physical regimes

The active PDE terms or their coefficients may vary across space and time, indicating different physical regimes. One can treat each space-time point as an observation and compare which terms are active locally. For example, a flow may be advection-dominated in one region and diffusion-dominated in another. This is not merely clustering the raw state u(x,t); the objective is to identify regimes associated with different governing terms.

## 6. Koopman Operators

A nonlinear dynamical system can be written as

$$
x_{k+1}=F(x_k).
$$

The Koopman-operator viewpoint shifts attention from the state x itself to observable functions of the state. An observable g maps a state to a quantity of interest. The Koopman operator K acts on observables by composition with the system dynamics:

$$
(\mathcal{K}g)(x)=g(F(x)).
$$

The operator K is linear in g, even when the state-transition map F is nonlinear. This does not make the original finite-dimensional dynamics linear in the original state coordinates. Instead, the dynamics are represented linearly in a space of observables, which may be infinite-dimensional.

This viewpoint makes tools from linear operator theory, including eigenvalues, eigenfunctions and spectral analysis, potentially useful. If φ is a Koopman eigenfunction with eigenvalue λ, then

$$
\mathcal{K}\phi=\lambda\phi,
$$

or, along a discrete trajectory,

$$
\phi(x_{k+1})=\lambda\phi(x_k).
$$

The transformed coordinate z_k = φ(x_k) consequently evolves linearly as z_{k+1} = λz_k for this eigenfunction. In practice, a computer cannot represent an arbitrary infinite-dimensional observable space exactly, so computational methods must approximate it with finite representations.

## 7. Reduced-Order Modelling (ROM)

Reduced-order modelling aims to reduce the cost of simulating a high-dimensional system. Suppose a spatial field u(x,t) is discretised at N locations. At each time t, it becomes a vector in R^N:

$$
\mathbf{u}(t)=[u(x_1,t),u(x_2,t),\ldots,u(x_N,t)]^{\mathsf T}.
$$

Although N may be very large, the observed dynamics may be well approximated by a low-dimensional subspace. ROM seeks a small collection of basis functions φ₁,…,φᵣ, with r much smaller than N, and represents the field as

$$
u(x,t)\approx \bar{u}(x)+\sum_{k=1}^{r}a_k(t)\phi_k(x).
$$

Here ū(x) is the mean field, φ_k(x) are spatial modes, and a_k(t) are time-dependent coefficients. The full high-dimensional field is then represented using a much smaller number of coordinates a_k(t). Proper orthogonal decomposition (POD) is a standard data-driven way to obtain spatial modes.

## 8. Snapshot Matrices, SVD and POD

A snapshot is the system state at a particular time. After spatial discretisation, a snapshot can be represented as an N-dimensional vector u_j. With M snapshots, first compute the mean state

$$
\bar{\mathbf{u}}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{u}_j,
$$

and centre the snapshots:

$$
\mathbf{q}_j=\mathbf{u}_j-\bar{\mathbf{u}}.
$$

Place the centred snapshot vectors into a matrix

$$
Q=[\mathbf{q}_1\;\mathbf{q}_2\;\cdots\;\mathbf{q}_M].
$$

The goal is to find a low-dimensional subspace that captures as much of the snapshot data as possible. Compute the singular value decomposition (SVD):

$$
Q=U\Sigma V^{\mathsf T}.
$$

The columns of U are orthonormal spatial modes. The singular values in Σ are conventionally ordered from largest to smallest; larger singular values indicate modes that contribute more to representing the snapshot matrix. Keeping the leading r modes gives a low-rank approximation.

$$
Q\approx U_r\Sigma_r V_r^{\mathsf T}.
$$

For a rank-one decomposition, Q can be written as a sum of outer products of left and right singular vectors weighted by singular values:

$$
Q=\sum_k \sigma_k\mathbf{u}_k\mathbf{v}_k^{\mathsf T}.
$$

The corresponding snapshot coefficients are obtained from the temporal/right-singular-vector factors, scaled by the singular values. Since UᵀU = I, the retained spatial modes are orthonormal. POD is analogous in spirit to a Fourier expansion, but its basis is learned from data rather than fixed in advance.

## 9. Galerkin Projection

After reducing the state dimension, one still needs equations for the reduced coefficients. Galerkin projection can turn a PDE into a lower-dimensional system of ODEs. Consider, for example, the PDE

$$
u_t=u_{xx}.
$$

Use the reduced expansion u(x,t) ≈ ū(x) + Σ_{k=1}^r a_k(t)φ_k(x). Substitution into the PDE generally leaves a residual R(x,t), defined as the difference between the time derivative and the spatial operator applied to the approximation. Rather than requiring the residual to vanish at every x, Galerkin projection requires it to be orthogonal to each retained basis function:

$$
\langle R(\cdot,t),\phi_i\rangle=0,\quad i=1,\ldots,r.
$$

For an orthonormal basis and a linear example, this produces equations for the coefficients. In a simplified derivation,

$$
\dot{a}_i=\sum_{k=1}^{r}a_k\langle\phi_k^{\prime\prime},\phi_i\rangle.
$$

Define A_ik = ⟨φ_k'', φ_i⟩. The reduced system becomes

$$
\dot{\mathbf{a}}=A\mathbf{a}.
$$

Once the reduced ODE is solved for a(t), the coefficients can be substituted into the reduced expansion to reconstruct an approximation of the field. The procedure trades some spatial detail for computational efficiency. For turbulent flows, many interacting scales may make a low-dimensional model inadequate unless unresolved effects are represented appropriately.

## 10. Operator Inversion for PDEs

A PDE can be expressed using an operator. For example, a linear, time-independent problem may be written

$$
\mathcal{L}u=f,
$$

where L is a differential operator, u is the unknown solution and f is the forcing. Formally, if an inverse operator exists under the specified boundary conditions,

$$
u=\mathcal{L}^{-1}f.
$$

For linear PDEs, Green's functions and eigenfunction expansions are two classical ways to represent solutions. The exact representation depends on the operator, its domain and boundary conditions. For nonlinear PDEs, superposition generally fails, making a linear inverse-operator representation unavailable in the same form.

## 11. Green's Functions

The Dirac delta δ is a distribution concentrated at one point. Informally, it is zero away from its support and has unit integral:

$$
\delta(x)=0\ \text{for }x\ne 0,\qquad \int_{-\infty}^{\infty}\delta(x)\,dx=1.
$$

Its defining sampling property is

$$
\int \delta(\xi-x)u(\xi)\,d\xi=u(x),
$$

under the appropriate distributional interpretation. A Green's function describes the response of a linear differential operator to a point source. One common convention defines G by

$$
\mathcal{L}_{\xi}G(\xi,x)=\delta(\xi-x),
$$

with boundary conditions appropriate to the problem. For non-self-adjoint operators, the adjoint operator may appear in the defining equation, depending on the convention and the inner-product identity being used.

The idea is that a general forcing function can be assembled from point sources. For a suitable linear problem, the solution can therefore be represented as an integral of the Green's function against the forcing term:

$$
u(x)=\int_{\Omega}G(x,\xi)f(\xi)\,d\xi.
$$

The variable x denotes the observation point, while ξ is the integration or source variable. The Green's function tells us how a source at one location contributes to the solution at another. It is an integral kernel, but it should not be confused with an arbitrary machine-learned kernel: the Green's function is tied to a specific differential operator and its boundary conditions.

A derivation often uses the adjoint identity

$$
\langle \mathcal{L}f,g\rangle=\langle f,\mathcal{L}^{*}g\rangle,
$$

together with the delta sampling property. Care is required with the placement of variables, the adjoint, and boundary terms; different conventions can write the Green's function arguments in different orders.

## 12. Eigenfunction Expansion

For an operator with a suitable eigenfunction basis, suppose

$$
\mathcal{L}\phi_n=\lambda_n\phi_n.
$$

If the eigenfunctions form a sufficiently complete orthogonal basis and the problem satisfies the relevant assumptions, expand the solution as

$$
u(x)=\sum_n b_n\phi_n(x).
$$

Substituting this expansion into Lu = f gives

$$
\sum_n b_n\lambda_n\phi_n=f.
$$

Taking the inner product with φ_m and using orthogonality yields

$$
\langle\phi_m,f\rangle=b_m\lambda_m\langle\phi_m,\phi_m\rangle.
$$

Hence, when λ_m is nonzero,

$$
b_m=\frac{\langle\phi_m,f\rangle}{\lambda_m\langle\phi_m,\phi_m\rangle}.
$$

If the eigenfunctions are normalised to unit norm, this simplifies to b_m = ⟨φ_m,f⟩/λ_m. The resulting expansion expresses the solution through the spectral properties of L. Completeness, boundary conditions and possible zero eigenvalues must be handled for the representation to be valid.

## 13. From Classical Inversion to Operator Learning

Classical Green's-function and eigenfunction methods exploit the structure of linear operators. For nonlinear operators, a comparable closed-form inverse is generally unavailable. A data-driven approach instead trains a parameterised model to approximate the mapping from the input of a PDE problem to its solution.

$$
\mathcal{G}_{\theta}(f)\approx u,
$$

where f represents an input such as a forcing term, coefficient field or initial condition. Operator learning targets a map between functions, rather than merely a map between finite-dimensional vectors. A training dataset can contain many input functions and their corresponding solution fields.

For a linear problem, an integral representation may involve a Green's function. In a learned neural operator, an integral kernel can be parameterised and trained, but it should not automatically be identified with a classical Green's function. The learned model approximates a solution operator, and its internal kernels generally need not have the same mathematical meaning as a Green's function.

## 14. Neural Operators

A representative neural-operator layer has the form

$$
v_{k+1}(x)=\sigma\left(W_kv_k(x)+\int_{D_k}K_{\theta}^{(k)}(x,y)v_k(y)\,dy+b_k(x)\right).
$$

Here v_k is a feature function at layer k, W_k is a pointwise linear transformation, K_θ is a parameterised kernel, b_k is a bias term, and σ is an activation function. The integral aggregates information from locations y across a domain, allowing the representation at x to depend on information elsewhere in the domain. This is a non-local component.

The activation function may be ReLU or another suitable nonlinear function. Training adjusts the parameters θ within the chosen architecture. The output is intended to represent a function, which is why neural operators are designed to learn mappings between function spaces.

In computation, functions must be sampled or discretised. A key design goal of neural operators is to support transfer across discretisations: the same learned operator can, under suitable architectural and numerical conditions, be evaluated on different grids. This is often described as discretisation invariance or mesh-free generalisation, but it should not be interpreted as a guarantee of identical accuracy on every mesh.

### 14.1 Fourier neural operators

Fourier Neural Operators (FNOs) parameterise integral interactions in the spectral domain. They use Fourier transforms to represent global interactions efficiently, often by learning transformations of selected Fourier modes. This can be effective for spatially structured PDE data, although performance depends on the problem, resolution and training distribution.

## 15. DeepONet

The universal approximation theorem for ordinary neural networks concerns approximation of functions under suitable conditions. Operator learning extends the goal to mappings from input functions to output functions. DeepONet is an operator-learning architecture built from a branch network and a trunk network.

- The branch network receives sampled values of the input function, such as u(x₁), …, u(x_m), and produces coefficients.

- The trunk network receives the query coordinate y and produces basis-like features at that location.

The two outputs are combined through an inner product. In a typical form,

$$
\mathcal{G}(u)(y)\approx\sum_{k=1}^{p}b_k(u)t_k(y).
$$

For example, if the branch output is [c₁,c₂,c₃] and the trunk output is [t₁(y),t₂(y),t₃(y)], the prediction is

$$
\mathcal{G}(u)(y)\approx c_1t_1(y)+c_2t_2(y)+c_3t_3(y).
$$

The branch network encodes the input function; the trunk network represents how the output is evaluated at a chosen coordinate. The model can then be queried at different output locations y. The structure resembles a basis expansion, with coefficients depending on the input function and basis-like features depending on the query location.

## 16. PINN: Physics-Informed Neural Networks

A physics-informed neural network (PINN) approximates a solution to a particular PDE using a neural network. For a space-time problem, the network takes coordinates (x,t) as input and returns an approximate solution:

$$
(x,t)\mapsto \mathrm{NN}\mapsto \hat{u}_{\theta}(x,t).
$$

Consider the illustrative advection equation

$$
u_t+u_x=0.
$$

Substitute the neural-network approximation into the PDE to define the residual

$$
r_{\theta}(x,t)=\frac{\partial\hat{u}_{\theta}}{\partial t}+\frac{\partial\hat{u}_{\theta}}{\partial x}.
$$

The physics loss penalises a nonzero residual at selected collocation points. Automatic differentiation computes the derivatives of the network output with respect to the input coordinates; parameter gradients for training are then obtained through differentiation of the total loss.

The PDE alone may admit multiple solutions, so initial and boundary conditions are also required. If u₀(x) is the initial condition and g describes the boundary data, a typical total objective is

$$
\mathcal{L}_{\mathrm{total}}=\lambda_r\mathcal{L}_{\mathrm{PDE}}+\lambda_0\mathcal{L}_{\mathrm{IC}}+\lambda_b\mathcal{L}_{\mathrm{BC}}.
$$

The weights λ_r, λ_0 and λ_b balance the residual, initial-condition and boundary-condition losses. Gradient-based optimisation adjusts the network parameters to reduce this objective.

### 16.1 PINN limitations

A low training loss does not by itself guarantee an accurate solution. Optimisation can be difficult for stiff or multiscale PDEs, and the balance among losses can matter substantially. Accuracy should be checked against analytical solutions, numerical solutions or independent data where available.

## 17. PINO: Physics-Informed Neural Operators

A physics-informed neural operator (PINO) combines operator learning with physics-informed constraints. The goal is not merely to approximate one solution at one set of coordinates, but to learn a solution operator that maps an initial condition or other input function to a future solution field:

$$
\mathcal{G}_t:u_0(\cdot)\mapsto u(\cdot,t).
$$

This distinction is useful when a family of related PDE problems has different initial conditions. Once trained, the operator can be applied to new inputs within its domain of generalisation.

A representative physics residual for an evolution equation is

$$
r_{\theta}=\frac{\partial u_{\theta}}{\partial t}+\mathcal{N}(u_{\theta}).
$$

PINO can combine a data loss, comparing predicted and reference solutions, with a physics loss penalising PDE residuals:

$$
\mathcal{L}_{\mathrm{data}}=\|\mathcal{G}_{\theta}(u_0)-u_{\mathrm{true}}\|^2,
$$

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{data}}+\lambda\mathcal{L}_{\mathrm{physics}}.
$$

The data term encourages agreement with observed or numerically generated solutions; the physics term encourages consistency with the governing equation. Automatic differentiation may be used to evaluate derivatives in the residual. Unlike a purely data-driven neural operator, a PINO explicitly incorporates a PDE-based constraint into training.

PINO is conceptually relevant to weather prediction because different initial atmospheric states lead to different future states. However, this does not mean that operational weather models such as Pangu-Weather are PINO models. Data-driven weather forecasting systems can use different architectures and training objectives.

## 18. PINN, PINO and Neural Operators: Key Distinctions

- PINN: learns a particular solution function u(x,t) for a specified PDE problem, using PDE residuals and conditions as constraints.

- Neural operator: learns a function-to-function mapping from inputs of a problem to its solution, commonly using paired input-output data.

- PINO: learns a solution operator while incorporating physics-based residual constraints, often alongside data-fitting loss.

These methods overlap in their use of neural networks, but their learning targets are different. The distinction is between learning a single solution and learning a reusable operator over a family of inputs, and between relying primarily on data and explicitly enforcing physical structure.

## 19. Overall Conceptual Map

The topics in these notes address different parts of the PDE modelling pipeline:

1. PDE discovery asks what governing equation may explain observed data.

1. Closure modelling addresses unresolved effects omitted by coarse-resolution models.

1. Koopman methods seek linear representations of nonlinear dynamics in observable space.

1. ROM, POD and SVD compress high-dimensional solution fields into low-dimensional representations.

1. Galerkin projection derives reduced dynamical equations for the retained coefficients.

1. Green's functions and eigenfunction expansions provide classical solution representations for suitable linear operators.

1. Neural operators and DeepONet learn solution mappings across families of PDE inputs.

1. PINN imposes PDE constraints while approximating a specific solution; PINO imposes such constraints while learning an operator.

These approaches are not interchangeable. The appropriate method depends on the objective: discovering equations, reducing simulation cost, approximating one solution, or learning a reusable mapping across many related PDE problems.

## 20. Notation and terminology

- PDE: partial differential equation.

- ODE: ordinary differential equation.

- IC / BC: initial condition / boundary condition.

- Governing equation: equation describing the evolution or constraints of a physical system.

- Closure term: modelled contribution of unresolved degrees of freedom to resolved dynamics.

- Snapshot: a sampled system state at a particular time.

- SVD: singular value decomposition.

- POD: proper orthogonal decomposition.

- ROM: reduced-order model or modelling.

- Residual: the discrepancy obtained when an approximate solution is substituted into an equation.

- Test function: a function used to formulate a weak/integral statement of a differential equation.

- Solution operator: a mapping from problem inputs, often functions, to solution functions.

- Kernel: a function used to express interactions or an integral mapping; its meaning depends on the mathematical context.

- Discretisation: replacing a continuous domain or function with a finite representation for computation.