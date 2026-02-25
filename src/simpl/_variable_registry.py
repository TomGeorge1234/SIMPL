"""Variable metadata registry for the SIMPL results dataset."""


def build_variable_info_dict(dim: list[str]) -> dict:
    """Build the dictionary describing every variable stored in SIMPL results.

    This dictionary summarises _all_ the variables used and returned by the
    SIMPL class.  Each entry is attached as the ``attrs`` to the DataArray when
    these variables are saved.  Each entry is a dict with:

        name : str — the name of the variable
        description : str — a brief description of the variable
        dims : list — the dimensions of the variable (excluding any epoch
            dimension)
        axis_title : str — the title of the axis when plotting the variable
        formula : str — a formula for the variable (if applicable)
        reshape : bool — whether to forcefully reshape to the dimensions given
            in ``dims`` when saved.  This is useful for variables calculated in
            a different shape to the final intended shape (e.g. receptive
            fields).

    Parameters
    ----------
    dim : list[str]
        Environment dimension names, e.g. ``['x', 'y']``.
    """
    variable_info_dict = {
        # Core data variables
        "Y": {
            "name": "Spikes",
            "description": (
                "The spikes of the neurons at each time step. Each a binary vector of length N_neurons."
            ),
            "dims": ["time", "neuron"],
            "axis_title": "Spike counts",
            "formula": r"$y(t)$",
        },
        "X": {
            "name": "Latent",
            "description": (
                "The latent positions of the agent at each "
                "time step. This is (typically) the smoothed "
                "output of the Kalman filter, scaled to "
                "correlate maximally with behaviour "
                "(X <-- mu_s, X <-- X @ coef.T + intercept). "
                "For epoch 0, X == the behaviour."
            ),
            "dims": ["time", "dim"],
            "axis_title": "Position",
            "formula": r"$x(t)$",
        },
        "F": {
            "name": "Model",
            "description": (
                "The receptive fields of the neurons "
                "(probability of the neurons firing at each "
                "position in one time step (of length dt)."
            ),
            "dims": ["neuron", *dim],
            "axis_title": "Receptive field",
            "reshape": True,
            "formula": r"$r(x)$",
        },
        "F_odd_minutes": {
            "name": "Model (odd minutes)",
            "description": (
                "The receptive fields of the neurons "
                "(probability of the neurons firing at each "
                "position in one time step) calculated from "
                "the odd minutes of the data."
            ),
            "dims": ["neuron", *dim],
            "axis_title": "Receptive field (odd mins)",
            "Formula": r"$r_{\textrm{odd}}(x)$",
            "reshape": True,
        },
        "F_even_minutes": {
            "name": "Model (even minutes)",
            "description": (
                "The receptive fields of the neurons "
                "(probability of the neurons firing at each "
                "position in one time step) calculated from "
                "the even minutes of the data."
            ),
            "dims": ["neuron", *dim],
            "axis_title": "Receptive field (even mins)",
            "Formula": r"$r_{\textrm{even}}(x)$",
            "reshape": True,
        },
        "FX": {
            "name": "Smoothed firing rates",
            "description": (
                "An estimate of the firing rate of the "
                "neurons at each time step based on the "
                "latest position estimates and their "
                "receptive fields."
            ),
            "dims": ["time", "neuron"],
            "axis_title": r"Firing rate",
            "formula": r"$f(t)$",
        },
        "PX": {
            "name": "Occupancy",
            "description": (
                "The spatial occupancy at each position bin, "
                "estimated from the latent trajectory using a "
                "kernel density estimator. This is the "
                "denominator of the KDE used to fit receptive "
                "fields."
            ),
            "dims": [*dim],
            "axis_title": "Occupancy",
            "formula": r"$p(x)$",
            "reshape": True,
        },
        # Ground truth data variables
        "Xb": {
            "name": "Position (behaviour)",
            "description": (
                "The position of the agent (i.e. not "
                'necessarily the "true" latent position which '
                "generated the spikes) at each time step. "
                "This is the behaviour of the agent and acts "
                "as the starting conditions for the algorithm "
                "X[epoch=0] == Xb."
            ),
            "dims": ["time", "dim"],
            "axis_title": "Position (behaviour)",
            "formula": r"$x_{\textrm{beh}}(t)$",
        },
        "Xt": {
            "name": "Latent (ground truth)",
            "description": (
                "The ground truth latent positions of the "
                "agent at each time step. If using real "
                "neural data, this is typically not available."
            ),
            "dims": ["time", "dim"],
            "axis_title": "Position (true)",
            "formula": r"$x_{\textrm{true}}(t)$",
        },
        "Ft": {
            "name": "Model (ground truth)",
            "description": (
                "The ground truth receptive fields. These are "
                "the true receptive fields of the neurons used "
                "to generate the data. If using real neural "
                "data, this is typically not available."
            ),
            "dims": ["neuron", *dim],
            "reshape": True,
            "axis_title": "Receptive field (true)",
            "formula": r"$r_{\textrm{true}}(x)$",
        },
        # Likelihood maps and Gaussian posterior parameters
        "logPYXF_maps": {
            "name": "Log-likelihoods",
            "description": (
                "The log-likelihood maps of the spikes, as a "
                "function of position, is calculated for each "
                "time step."
            ),
            "dims": ["time", *dim],
            "axis_title": "Log-likelihood map",
            "formula": r"$\log P(Y|x, \Theta)$",
            "reshape": True,
        },
        "mu_l": {
            "name": "Likelihood mean",
            "description": "The mean of the Gaussian fitted to the log-likelihood maps.",
            "dims": ["time", "dim"],
            "axis_title": "Likelihood mean",
            "formula": r"$\mu_l(t)$",
        },
        "mode_l": {
            "name": "Likelihood mode",
            "description": "The mode of the Gaussian fitted to the log-likelihood maps.",
            "dims": ["time", "dim"],
            "axis_title": "Likelihood mode",
            "formula": r"$\textrm{Mo}(t)$",
        },
        "sigma_l": {
            "name": "Likelihood covariance",
            "description": "The covariance matrix of the Gaussian fitted to the log-likelihood maps.",
            "dims": ["time", "dim", "dim_"],
            "axis_title": "Likelihood covariance",
            "formula": r"$\Sigma_l(t)$",
        },
        "mu_f": {
            "name": "Kalman filtered mean",
            "description": "The mean of the Kalman filtered posterior of the latent positions.",
            "dims": ["time", "dim"],
            "axis_title": "Kalman filtered mean",
            "formula": r"$\mu_f(t)$",
        },
        "sigma_f": {
            "name": "Kalman filtered covariance",
            "description": "The covariance matrix of the Kalman filtered posterior of the latent positions.",
            "dims": ["time", "dim", "dim_"],
            "axis_title": "Kalman filtered covariance",
            "formula": r"$\Sigma_f(t)$",
        },
        "mu_s": {
            "name": "Kalman smoothed mean",
            "description": "The mean of the Kalman smoothed posterior of the latent positions.",
            "dims": ["time", "dim"],
            "axis_title": "Kalman smoothed mean",
            "formula": r"$\mu_s(t)$",
        },
        "sigma_s": {
            "name": "Kalman smoothed covariance",
            "description": "The covariance matrix of the Kalman smoothed posterior of the latent positions.",
            "dims": ["time", "dim", "dim_"],
            "axis_title": "Kalman smoothed covariance",
            "formula": r"$\Sigma_s(t)$",
        },
        # Linear scaling parameters
        "coef": {
            "name": "Linear scaling coefficients",
            "description": (
                "The linear scaling matrix coefficients "
                "applied, wlog, to the latent positions to "
                "maximally correlate it with the behaviour."
            ),
            "dims": ["dim", "dim_"],
            "axis title": "Linear scaling coefficients",
            "formula": r"$\mathbf{M}$",
        },
        "intercept": {
            "name": "Linear scaling intercept",
            "description": (
                "The linear scaling intercept vector applied, "
                "wlog, to the latent positions to maximally "
                "correlate with the behaviour."
            ),
            "dims": ["dim"],
            "axis title": "Linear scaling intercept",
            "formula": r"$\mathbf{c}$",
        },
        # Place field stuff
        "place_field_count": {
            "name": "Number of place fields indentified",
            "description": (
                "The number of place fields identified per "
                "neuron. Place fields are defined by "
                "thresholding the tuning curves (2D only) at "
                "1 Hz and identifying continuous regions which "
                "are both (i) under one-third the full size of "
                "the environment and (ii) have a maximum "
                "firing rate over 2 Hz."
            ),
            "dims": ["neuron"],
            "formula": r"$N_{pf}$",
            "axis title": "Number of place fields",
        },
        "place_field_position": {
            "name": "Place field position",
            "description": "The mean of the Gaussian fitted to each place field.",
            "dims": ["neuron", "place_field", "dim"],
            "axis title": "Place field position",
            "formula": r"$\mu_{pf}$",
        },
        "place_field_covariance": {
            "name": "Place field covariance",
            "description": "The covariance of the Gaussian fitted to each place field",
            "dims": ["neuron", "place_field", "dim", "dim_"],
            "axis title": "Place field covariance",
            "formula": r"$\Sigma_{pf}$",
        },
        "place_field_size": {
            "name": "Place field size",
            "description": "The size of the place field (units of m^2)",
            "dims": ["neuron", "place_field"],
            "axis title": "Place field size",
            "formula": r"$S_{pf}$",
        },
        "place_field_outlines": {
            "name": "Place field outlines",
            "description": "The outlines of the identified place fields (if any)",
            "dims": ["neuron", *dim],
            "axis title": "Place field outlines",
        },
        "place_field_roundness": {
            "name": "Place field roundness",
            "description": (
                "The roundness of the identified place fields (if any). Defined as r = 4 * pi * area / perimeter^2"
            ),
            "dims": ["neuron", "place_field"],
            "axis title": "Place field roundness",
        },
        "place_field_max_firing_rate": {
            "name": "Maximum firing rate of the place field",
            "description": (
                "Maximum firing rate -- in units of spikes per time bin rather than Hz -- of the place field"
            ),
            "dims": ["neuron", "place_field"],
            "axis title": "Place field max. firing rate",
        },
        # Result metrics
        "logPYF": {
            "name": "Total data log-likelihood",
            "description": (
                "Suppose we have two sets of observations, "
                "Y and Y_prime. The log-likelihood of the "
                "observations := P(Y | Y_prime) can be found "
                "by using Y_prime to estimate the latent "
                "posterior P(X | Y_prime) (i.e. run the "
                "Kalman model) and then marginalise out X "
                "from the likelihood of the new observations "
                "given the latent: P(Y | Y_prime) = int_X "
                "P(Y | X) P(X | Y_prime). This is called the "
                "posterior predictive and, for Kalman models, "
                "has analytic form P(Y_i | Y_prime) = "
                "Normal(Y_i | Y_hat, S) where "
                "S = H @ sigma_i @ H.T + R (the posterior "
                "observation covariance combined with the "
                "observation noise covariance) and "
                "Y_hat = H @ mu_i (the predicted "
                "observation), see page 361 of the Advanced "
                "Murphy book. Strictly this metric (i) "
                "returns not the full distribution (a product "
                "of many Gaussians) but the probability "
                "density evaluated at the observation "
                "locations and (ii) uses the same observations "
                "for both sets Y = Y_prime = "
                "observations_from_training_spikes. It is a "
                "good gauge on the performance of the model "
                '"were the observations (~spikes) P(Y|X) '
                "likely under the decoded trajectory "
                'P(X|Y_prime)?" but note that Y are the '
                "observations not the actual spikes so this "
                "is not strictly the likelihood of the spikes."
            ),
            "dims": [],
            "axis title": "Data log-likelihood",
            "formula": r"$\log P(Y|Y_{\textrm{train}})$",
        },
        "logPYF_test": {
            "name": "Total data log-likelihood (test)",
            "description": (
                "See logPYF but for observations derived from "
                "the testing spikes i.e. logPYF_test = "
                "P(Y_test|Y_train) = int_X "
                "P(Y_test|X)P(X|Y_train). Its analogous to "
                'asking "were the test observations (~test '
                "spikes) P(Y_test|X) likely under the decoded "
                'trajectory P(X|Y_train)?" or in other words '
                '"can the train spikes be used to predict the '
                'test spikes?"'
            ),
            "dims": [],
            "axis title": "Data log-likelihood (test)",
            "formula": r"$\log P(Y_{\textrm{test}}|Y_{\textrm{train}})$",
        },
        "logPYXF": {
            "name": "Mean spike log-likelihood given trajectory",
            "description": (
                "This is the poisson log-likelihood of the "
                "spikes along the trajectory. It is different "
                "to the logPYF metric in that it doesnt "
                "account for the posterior uncertainty in X "
                "and is a direct reading of the spike "
                "likelihood rather than the likelihood of the "
                "observations of those spikes (likelihood map "
                "modes). It is normalised by the number of "
                "spike-time bins to make it comparable across "
                "test and train datasets."
            ),
            "dims": [],
            "axis title": "Spike log-likelihood",
            "formula": r"$\log P(Y|X(t), \Theta)$",
        },
        "logPYXF_test": {
            "name": "Mean spike log-likelihood given trajectory (test)",
            "description": (
                "See logPYXF but for observations derived "
                "from the testing spikes. This is the poisson "
                "log-likelihood of the test spikes along the "
                "trajectory derived from the train spikes."
            ),
            "dims": [],
            "axis title": "Spike log-likelihood (test)",
            "formula": r"$\log P(Y_{\textrm{test}}|X(t), \Theta)$",
        },
        "spatial_information": {
            "axis title": "Spatial Information (bits/spike)",
            "name": "Spatial information",
            "description": (
                "Following 'An Information-Theoretic "
                "Approach to Deciphering the Hippocampal "
                "Code', the formula to compute the spatial "
                "information is as follows: "
                r"$I=\int_x \lambda(x) \log _2 "
                r"\frac{\lambda(x)}{\lambda} p(x) d x$ "
                r"where $\lambda(x)$ is the place field of "
                r"the cell, $\lambda$ is the mean firing "
                "rate of the cell, and $p(x)$ is the "
                "spatial occupancy. "
                "https://proceedings.neurips.cc/paper/1992/"
                "file/5dd9db5e033da9c6fb5ba83c7a7ebea9-"
                "Paper.pdf"
            ),
            "dims": ["neuron"],
            "formula": r"$I=\int_x \lambda(x) \log _2 \frac{\lambda(x)}{\lambda} p(x) d x$",
        },
        "spatial_information_rate": {
            "axis title": "Spatial Information Rate (bits/s)",
            "name": "Spatial information rate",
            "description": (
                "How many bits of information per second the "
                "spikes give us (on average) about the agent's "
                "position. Computed as the sum of per-neuron "
                "Skaggs spatial information rates."
            ),
            "dims": [],
            "formula": r"$\dot{\mathcal{I}} = \sum_n I_n$",
        },
        "negative_entropy": {
            "name": "Negative entropy",
            "description": "The negative entropy of each of the receptive fields, -sum(F * log(F)).",
            "dims": ["neuron"],
            "axis title": "Negative entropy",
            "formula": r"$- \sum F \log F$",
        },
        "sparsity": {
            "name": "Sparsity",
            "description": "The fraction of bins where the firing is greater than 0.1 * max firing rate.",
            "dims": ["neuron"],
            "axis title": "Spatial sparsity",
            "formula": r"$\rho(r(x))$",
        },
        "stability": {
            "name": "Stability",
            "description": (
                "The correlation between receptive fields "
                "estimated separately using spikes from odd "
                "and even minutes."
            ),
            "dims": ["neuron"],
            "axis title": "Field stability",
            "formula": r"$\textrm{Corr}(r_{\textrm{odd}}(x), r_{\textrm{even}}(x))$",
        },
        "field_count": {
            "name": "Number of fields",
            "description": (
                "A heuristic on the number of distinct fields "
                "in the place fields calculated as the number "
                "of connected components in the receptive "
                "fields thresholded at 0.5 * their maximum "
                "firing rate."
            ),
            "dims": ["neuron"],
            "axis title": "Number of fields",
            "formula": r"$N_{\textrm{fields}}$",
        },
        "field_size": {
            "name": "Field size",
            "description": "The average size of the fields in the individual fields, as defined in field_count.",
            "dims": ["neuron"],
            "axis title": "Field size",
            "formula": r"",
        },
        "field_change": {
            "name": "Field shift",
            "description": "The change in the place fields from the last epoch.",
            "dims": ["neuron"],
            "axis title": "Field change",
            "formula": r"$\|r_{e}(x) - r_{e-1}(x)\|$",
        },
        "trajectory_change": {
            "name": "Latent shift",
            "description": "The average change in the latent positions from the last epoch.",
            "dims": ["time"],
            "axis title": "Latent change",
            "formula": r"$\|x_{e}(t) - x_{e-1}(t)\|$",
        },
        # Baselines : calculated when ground truth data is available
        "X_R2": {
            "name": "Latent R2",
            "description": "The R2 score between the true and estimated latent positions.",
            "dims": [],
            "axis title": "Latent R2",
            "formula": r"$R^2$",
        },
        "X_err": {
            "name": "Latent error",
            "description": "The mean squared error between the true and estimated latent positions.",
            "dims": [],
            "axis title": "Latent error",
            "formula": r"$\|x_{\textrm{true}}(t) - x_{\textrm{est}}(t)\|$",
        },
        "F_err": {
            "name": "Model error",
            "description": "The mean squared error between the true and estimated place fields.",
            "dims": [],
            "axis title": "Receptive field error",
            "formula": r"$\|r_{\textrm{true}}(x) - r_{\textrm{est}}(x)\|$",
        },
        # Other variables
        "spike_mask": {
            "name": "Spike mask",
            "description": (
                "The mask used to determine which spikes are "
                "used for training and testing. This is "
                "usually a speckled mask (see Williams et al. "
                "2020, fig. 2) of mostly `True` interspersed "
                "with contiguous blocks of `False`. Where "
                "False, spikes are masked and NOT used for "
                "training."
            ),
            "dims": ["time", "neuron"],
            "axis_title": "Spike mask",
            "formula": r"$\textrm{mask}(t, n)$",
        },
    }
    return variable_info_dict
