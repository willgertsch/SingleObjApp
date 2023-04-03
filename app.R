#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

# I put everything into one file for easier deployment
# code is sourced from my main project

# required libraries
library(shiny)
library(ggplot2)
library(waiter)
library(metaheuristicOpt)

################################################################################
# Global variables for app
################################################################################
models = c(
  "Logistic",
  "Logistic quadratic",
  "Logistic cubic",
  "Logistic fractional polynomial",
  "Weibull",
  "Log-logistic",
  "Mixture multistage",
  "Box-Cox Weibull"
)

objectives = c(
  "D",
  "A"
)

algorithms = c(
  "PSO",
  "DE",
  "GWO",
  "HS"
)

bmd_models = c(
  "Logistic",
  "Weibull",
  "Log-logistic"
)

################################################################################
# UI and server functions for app
################################################################################

ui = fixedPage(
  withMathJax(),
  tabsetPanel(
    selected = "Find optimal designs",
    type = "tabs",
    id = "toptabpanel",
    tabPanel(
      "User manual",
      tags$h3("Overview",
              style = "text-align:left;"),
      tags$p(
        "This app allows the user to find optimal experimental designs for several
          nonlinear models used in toxicology. It also provides a tool for
          comparing the efficiency of any two designs. This page explains how to use
          the app and the optimal design theory used to generate and interpret the results."
      ),
      tags$p(
        "This app uses metaheuristic algorithms, which are a class of
          optimization methods that have been increasingly used in recent years
          to solve complex problems that are difficult to solve using standard
          methods. They are inspired by natural phenomena and simulate the
          behavior of a group of entities, such as animals, insects, or particles,
          in search of an optimal solution. Some examples of metaheuristic algorithms
          are Particle Swarm Optimization (PSO), Differential Evolution (DE),
          Harmony Search (HS), and Grey Wolf Optimizer (GWO). These algorithms have
          shown great success in solving a wide range of problems, including engineering,
          economics, logistics, and data analysis. They are especially useful in optimal
          design of experiments because they can easily be applied to a wide variety
          of design problems."
      ),
      tags$h3("The design tab",
              style = "text-align:left;"),
      tags$p(
        "The design tab allows the user to use metaheuristic algorithms to find
          the optimal design for a selected nonlinear model. The sidebar allows
          the user to set several algorithm options. The algorithm selector chooses
          metaheuristic to use to find the design and the iterations and swarm size
          options tell the algorithm how long to run and the diversity of the search.
          Increasing both the number of iterations and the swarm size will make it
          more likely that the optimal design will be found but increases the run
          time. The dose limit option contols the highest dose allowed.
          The limit should be chosen based on background knowledge of the
          experiment. A lower dose limit will limit the search space and make the
          optimal design easier to find. The design points option controls how
          many doses will be allowed in the optimal design. If too few doses are
          specified, then the optimal design might be impossible to find. A general
          rule of thumb is that the number of doses in the optimal design should
          be the same as the number of parameters in the model."
      ),
      tags$p(
        "The main pane alloes the user to select the model, parameter values,
          and the design objective. Theta values should be entered separated by
          commas. Pressing the find design button will run the selected algorithm
          options on the design problem, display the final design, and display a
          graphical check of optimality.
          If plotted function is less than 0 with equality at the doses in the design,
          then the optimal design has been found. If the design is not optimal,
          try rerunning with more iterations and a larger swarm size. A different
          algorithm might also help to find the optimal design. If the plot displays
          a message about a singular information matrix, then there is likely a
          problem with how the design problem is specified. The most common issue
          is that the number of doses is too few. Another common issue is that
          the dose range may be too small for the model in question."
      ),
      tags$h3("Compare tab",
              style = "text-align:left;"),
      tags$p(
        "The compare tab allows the user to compare two designs to see which
          is more efficient. This is especially useful when the goal is to compare
          a naive or more practical design against the optimal design.
          Design 1 is compared relative to design design 2.
          If design 1 is less efficient, then the efficiency will be less than 1.
          If design 1 is more efficient, then the efficiency will be greater than 1.
          Parameter values, dose levels, and weights should be entered as comma
          separated values."
      ),
      tags$h3("Background",
              style = "text-align:left;"),
      tags$p(
        "
          Suppose \\(Y\\) is an outcome variable of interest and \\(d\\) is the dosage or
          concentration of the substance under investigation. For this work, we
          additionally assumed that \\(Y\\) is a binary variable that is equal to one
          if toxicity has occured and equal to 0 otherwise. We also assume the
          that the dose \\(d\\) is in the range \\([0, D]\\) where \\(D\\) is highest
          experimentally feasible dose. The goal of a dose response experiment
          is to establish a mathematical relationship of the form
$$
P(Y = 1| d) = P(d) = f(d, \\theta) + \\epsilon
$$
where \\(f\\) is a possibly nonlinear function of the dose and parameters \\(\\theta\\).
The error term \\(\\epsilon\\) is most commonly assumed to have a normal distribution,
\\(N(0, \\sigma^2)\\). Common forms of \\(f\\) include the exponential and inverse logit functions.

The goal of optimal design is to choose good values for the doses included in the
experiment and the number of subjects to assign to each dose. Let \\(d_i\\) denote the
ith dose in the design and let \\(n_i\\) be the number of subjects assigned to the ith
dose for \\(i = 1, \\dots, k\\).  For generalizabillity and theoretical convenience,
we assume a fixed sample size \\(N\\) and work with the proportions assigned to each
dose instead of the raw \\(n_i\\). Define \\(w_i\\) as \\(w_i = n_i/N\\) such that \\(\\sum_i^k w_i = 1\\).
Using this weight-based notation, we can define a \\(k\\) dose design as a probability measure on \\([0, D]\\).
$$
\\xi = \\begin{pmatrix}
d_1, & \\dots, & d_k\\\
w_1, & \\dots, & w_k
\\end{pmatrix}
$$

The design \\(\\xi\\) is a compact representation of all the parameters that must be
chosen optimally in order to maximize some objective function.
The most common objective functions in optimal design are based on the model information matrix,
\\(M(\\xi, \\theta)\\). This information matrix can be thought of the potential observed
information matrix if the experiment is run using the design \\(\\xi\\). Note that \\(M(\\xi, \\theta)\\)
also depends on the parameter values of the model, meaning that a prior value for
the parameters must be supplied. This value can come from previous data or theoretical
models. The designs generated are referred to as \\textit{locally optimal} with
respect to the prior \\(\\theta\\).

For the nonlinear models under consideration, the information matrix can be written as
$$
M = M(\\xi, \\theta) = \\sum_{i=1}^k w_i M_i(\\xi, \\theta) = \\sum_{i=1}^k w_i \\nabla f(d_i, \\theta) (\\nabla f(d_i, \\theta))'
$$
Note that this method of writing the information matrix decomposes the information
into the sum of the information contributed at each dose level.
The last equality says that the information matrix is a function of the gradient
with respect to the model parameters. This result follows from a linearization of
the nonlinear model function using the Delta method."
      ),
      tags$p(
        "Objective functions for optimal design are based around minimizing the
          variance estimated quantities or, equivalently, maximizing information
          gain of the experiment. For example, a common design strategy is to maximize
$$
\\Psi_D(M) = \\log |M|
$$
Maximizing \\(\\Psi_D\\) is equivalent to minimizing the size of the confidence
region of the parameter estimates and a design that minimizes \\(\\Psi_D\\) is called
D-optimal. Another approach is to minimize the objective function
$$
\\Psi_A(M) = \\operatorname{tr} M^{-1}
$$
which is equivalent to minimizing the sum of the variances of the parameter estimates.
Another useful objective function is
$$
\\Psi_c(M) = c'M^{-1}c
$$
which is minimized in order to minimize the variance of some linear combination
of the parameters. The reason why \\(\\Psi_c\\) is so useful is because it can be
applied to construct a wide variety of domain specific objectives. Suppose \\(g(d, \\theta)\\)
is some quantity of interest to be estimated. The variance of the estimated \\(g(d, \\theta)\\)
can be approximated using the Delta method as
$$
\\operatorname{Var}(g(d, \\hat\\theta)) = \\nabla g(d, \\hat\\theta)' M^{-1} \\nabla g(d, \\hat \\theta)
$$
This means that for an arbitrary function \\(g\\), an objective function for finding
the relevant optimal design can be found using the framework of c-optimality.

 In order to make comparisons between different designs, it is useful to consider
 efficiency relative to the optimal design. For example, we may wish to see how a
 naive design compares to the optimal design. In the case of D-optimality,
 let \\(\\xi_D\\) be the D-optimal design. The D-efficiency of a design \\(\\xi\\) is defined as
 $$
 \\operatorname{eff}_D(\\xi) = \\left(\\frac{|\\Psi_D(\\xi)|}{|\\Psi_D(\\xi_D)|}\\right)^{1/p}
 $$
 where \\(p\\) is the number of parameters in the model. Similar efficiency functions
 may be derived for the other optimality criteria.

 A final useful tool optimal design is the equivalence theorem of Kiefer (1960).
 This theorem says that if a design \\(\\xi\\) is optimal, then the directional
 derivative of the design criterion evaluated at \\(\\xi\\) must satisfy an inequality
 for all dose values in the design space with equality attained when the dose is
 in the design. This provides an easy graphical test to check if the design is
 optimal. These inequalities are well known for the most common design criteria and
 can easily be derived using matrix calculus for more uncommon criteria.
"
      )
    ),
    tabPanel(
      "Find optimal designs",

      tabsetPanel(
        selected = "Design",
        type = "pills",
        id = "designtabpanel",
        tabPanel("Design",
                 # sidebar layout for algorithm options
                 sidebarLayout(
                   sidebarPanel(
                     "Algorithm options",
                     selectInput("algorithm",
                                 "Algorithm",
                                 algorithms),
                     numericInput("iter",
                                  "Iterations",
                                  200,
                                  1,
                                  Inf,
                                  10),
                     numericInput("swarm",
                                  "Swarm size",
                                  30,
                                  10,
                                  Inf,
                                  10),
                     numericInput("bound",
                                  "Dose limit",
                                  10,
                                  0.1,
                                  Inf,
                                  1),
                     numericInput("pts",
                                  "Design points",
                                  2,
                                  1,
                                  10,
                                  1),
                     numericInput("seed",
                                  "Seed",
                                  155,
                                  1,
                                  Inf,
                                  1)

                   ),
                   mainPanel(
                     fluidRow(
                       column(
                         6,
                         selectInput("model_selector", "Model", models)
                       ),
                       column(
                         6,
                         textInput("theta_input", "Theta ( enter values separated by , )"),
                         selectInput("objective", "Objective", objectives)
                       )
                     ),
                     uiOutput("model_formula_display"),

                     actionButton("find", "Find design"),
                     plotOutput("sens_plot"),
                     waiter::use_waiter(),
                     verbatimTextOutput("design_out")
                   )
                 )),
        tabPanel(
          "Compare",
          "Compare the design efficiency of design \\(\\xi_1\\) relative to the design \\(\\xi_2\\).",
          selectInput("model_selector_compare", "Model", models),
          textInput("theta_input_compare", "Theta ( enter values separated by , )"),
          selectInput("objective_compare", "Objective", objectives),
          fluidRow(
            column(
              6,
              textInput("\\xi1_doses", "\\(\\xi_1 doses\\)"),
              textInput("\\xi1_weights", "\\(\\xi_1 weights\\)")
            ),
            column(
              6,
              textInput("\\xi2_doses", "\\(\\xi_2 doses\\)"),
              textInput("\\xi2_weights", "\\(\\xi_2 weights\\)")
            ),
            actionButton("compute_eff", "Compute efficiency"),
            textOutput("eff_out")
          )
        ),
        tabPanel("BMD Designs",
                 # sidebar layout for algorithm options
                 sidebarLayout(
                   sidebarPanel(
                     "Algorithm options",
                     selectInput("algorithm_bmd",
                                 "Algorithm",
                                 algorithms),
                     numericInput("iter_bmd",
                                  "Iterations",
                                  200,
                                  1,
                                  Inf,
                                  10),
                     numericInput("swarm_bmd",
                                  "Swarm size",
                                  30,
                                  10,
                                  Inf,
                                  10),
                     numericInput("bound_bmd",
                                  "Dose limit",
                                  10,
                                  0.1,
                                  Inf,
                                  1),
                     numericInput("pts_bmd",
                                  "Design points",
                                  2,
                                  1,
                                  10,
                                  1),
                     numericInput("seed_bmd",
                                  "Seed",
                                  155,
                                  1,
                                  Inf,
                                  1)

                   ),
                   mainPanel(
                     fluidRow(
                       column(
                         6,
                         selectInput("model_selector_bmd", "Model", bmd_models),
                         selectInput("risk_type_selector", "Risk type", c("Added", "Extra")),
                         numericInput("risk", "Risk increase",
                                      value = 0.1, min = .01, max = .99, step = 0.01)
                       ),
                       column(
                         6,
                         textInput("theta_input_bmd", "Theta ( enter values separated by , )"),
                         numericInput("lambda_input", "\\(\\lambda\\)",
                                      value = 0.5, min = 0, max = 1, step = 0.1)
                       )
                     ),
                     uiOutput("model_formula_display_bmd"),

                     actionButton("find_bmd", "Find design"),
                     plotOutput("sens_plot_bmd"),
                     waiter::use_waiter(),
                     verbatimTextOutput("design_out_bmd")
                   )
                 ))
      )


    )
  )
)

server = function(input, output, session) {

  # reactive data structure
  values <- reactiveValues()
  values$upload_state = NULL # flag for managing file uploads
  values$DT <- data.frame(x = numeric(),
                          y = numeric(),
                          yhat = numeric()
                          #color = factor(),
                          #shape = factor()
  )
  values$eff_crit = c("NA", "?")

  # set up reactive data structure
  # initialize with empty arrays and plots
  values$OD <- list(
    design = numeric(),
    sens_plot = ggplot2::ggplot(),
    msg = character()
  )

  # display model formula
  output$model_formula_display = renderUI({
    p(withMathJax(model_display(input$model_selector)))
  })

  # sensitivity plot
  output$sens_plot = renderPlot({

    # load plot from reactive data
    ggp = values$OD$sens_plot

    # display plot
    ggp
  })

  # action for Find button
  observeEvent(
    input$find,
    {
      # set up loading indicator
      waiter <- waiter::Waiter$new(
        id = "sens_plot",
        html = waiter::spin_terminal(),
        color = "grey"
      )$show()
      waiter$show()
      on.exit(waiter$hide())

      # grab and process theta from raw input
      theta = process_theta(input$theta_input)


      # select gradient function
      model = input$model_selector
      if (model == "Logistic")
        grad_fun = grad.logistic
      else if (model == "Logistic quadratic")
        grad_fun = grad.logistic.quad
      else if (model == "Logistic cubic")
        grad_fun = grad.logistic.cubic
      else if (model == "Logistic fractional polynomial")
        grad_fun = grad.logistic.fp
      else if (model == "Weibull")
        grad_fun = grad.weibull
      else if (model == "Log-logistic")
        grad_fun = grad.loglogistic
      else if (model == "Mixture multistage")
        grad_fun = grad.mix2
      else if (model == "Box-Cox Weibull")
        grad_fun = grad.boxcoxweibull

      # find optimal design
      out = find_design_single(
        grad_fun,
        input$objective,
        theta,
        input$bound,
        input$pts,
        input$algorithm,
        input$swarm,
        input$iter,
        input$seed
      )

      # update reactive data with new design data
      values$OD$msg = ""
      values$OD$design = out$result$result
      values$OD$sens_plot = out$plot
      #values$OD$response_plot = response_plot
      values$OD$val = out$result$optimumValue

    }
  )

  # design output
  output$design_out = renderPrint({

    raw = values$OD$design
    obj_val = values$OD$val[[1]]


    # case if algorithm hasn't run
    if (length(raw) == 0) {
      cat("No design")
    }
    else {
      cat("Objective value:", obj_val, "\n")

      # label and reorder
      l = length(raw)
      x = raw[1:(l/2)]
      w = raw[(l/2 + 1):l]
      cat("Doses:\n", x[order(x)], "\n", sep = " ")
      cat("Weights:\n", w[order(x)], "\n", sep = " ")
    }
  })

  ##############################################################################
  # compare designs tab
  ##############################################################################

  # run when compare designs button
  observeEvent(
    input$compute_eff,
    {

      d1 = process_theta(input$`\\xi1_doses`)
      d2 = process_theta(input$`\\xi2_doses`)
      w1 = process_theta(input$`\\xi1_weights`)
      w2 = process_theta(input$`\\xi2_weights`)
      theta = process_theta(input$theta_input_compare)
      eff = compute_eff(
        input$model_selector_compare,
        theta,
        input$objective_compare,
        d1,
        d2,
        w1,
        w2
      )

      # save eff to reactive data
      values$eff_crit = c(as.character(eff), input$objective_compare)

    }
  )

  # display design efficiency
  output$eff_out = renderText({
    sprintf("The %s-efficiency of design 1 relative to design 2 is %s",
            values$eff_crit[2], values$eff_crit[1])
  })

  ##############################################################################
  # BMD design tab
  # set up reactive data structure
  # initialize with empty arrays and plots
  values$OD2 <- list(
    design = numeric(),
    sens_plot = ggplot2::ggplot(),
    msg = character()
  )

  # display model formula
  output$model_formula_display_bmd = renderUI({
    p(withMathJax(model_display(input$model_selector_bmd)))
  })

  # sensitivity plot
  output$sens_plot_bmd = renderPlot({

    # load plot from reactive data
    ggp = values$OD2$sens_plot

    # display plot
    ggp
  })

  # action for Find button
  observeEvent(
    input$find_bmd,
    {
      # set up loading indicator
      waiter <- waiter::Waiter$new(
        id = "sens_plot_bmd",
        html = waiter::spin_terminal(),
        color = "grey"
      )$show()
      waiter$show()
      on.exit(waiter$hide())

      # grab and process theta from raw input
      theta = process_theta(input$theta_input_bmd)


      # select gradient function
      model = input$model_selector_bmd
      if (model == "Logistic")
        grad_fun = grad.logistic
      else if (model == "Weibull")
        grad_fun = grad.weibull
      else if (model == "Log-logistic")
        grad_fun = grad.loglogistic


      # find optimal design
      out = find_bmd_design(
        model,
        input$lambda_input,
        input$risk,
        input$risk_type_selector,
        theta,
        input$bound_bmd,
        input$pts_bmd,
        input$algorithm_bmd,
        input$swarm_bmd,
        input$iter_bmd,
        input$seed_bmd
      )


      # update reactive data with new design data
      values$OD2$msg = ""
      values$OD2$design = out$result$result
      values$OD2$sens_plot = out$plot
      #values$OD$response_plot = response_plot
      values$OD2$val = out$result$optimumValue
    }
  )

  # design output
  output$design_out_bmd = renderPrint({

    raw = values$OD2$design
    obj_val = values$OD2$val[[1]]


    # case if algorithm hasn't run
    if (length(raw) == 0) {
      cat("No design")
    }
    else {
      cat("Objective value:", obj_val, "\n")

      # label and reorder
      l = length(raw)
      x = raw[1:(l/2)]
      w = raw[(l/2 + 1):l]
      cat("Doses:\n", x[order(x)], "\n", sep = " ")
      cat("Weights:\n", w[order(x)], "\n", sep = " ")
    }
  })

}

################################################################################
# Gradient functions, objective functions, and derivatives
################################################################################
grad.logistic = function(x, theta) {

  eta = theta[1] + theta[2] * x
  sigma = exp(eta)/(1 + exp(eta))^2
  grad = sigma * c(1, x)
  return(grad)
}

grad.logistic.quad = function(x, theta) {

  eta = theta[1] + theta[2] * x + theta[3] * x^2
  sigma = exp(eta)/(1 + exp(eta))^2
  grad = sigma * c(1, x, x^2)
  return(grad)
}

grad.logistic.cubic = function(x, theta) {
  eta = theta[1] + theta[2] * x + theta[3] * x^2 + theta[4] * x^3
  sigma = exp(eta)/(1 + exp(eta))^2
  grad = sigma * c(1, x, x^2, x^3)
  return(grad)
}

# 2nd degree fractional polynomial predictor
grad.logistic.fp = function(x, theta) {

  # theta4 and theta5 are power paramters in this model
  powers = c(0, theta[4], theta[5])

  # x1 is the 2nd term in the polynomial
  x1 = H(2, x, powers)
  x2 = H(3, x, powers)
  eta = theta[1] + theta[2] * x1 + theta[3] * x2
  sigma = exp(eta)/(1 + exp(eta))^2
  grad = sigma * c(1, x1, x2)
  return(grad)

}

# mixture of two multistage models
# Razzaghi (2002) in Envirometrics
grad.mix2 = function(x, theta) {

  # identify parameters
  a = theta[1]
  b = theta[2]
  c = theta[3]
  d = theta[4]
  f = theta[5]
  g = theta[6]

  # gradient components
  d1 = g*exp(-a-b*x-c*x^2) + (1-g)*exp(-a-d*x-f*x^2)
  d2 = g*x*exp(-a-x*(b+c*x))
  d3 = g * x^2 * exp(-a-x*(b+c*x))
  d4 = (1 - g)*x*exp(-a-x*(d+f*x))
  d5 = (1 - g) *x^2 * exp(-a-x*(d+f*x))
  d6 = exp(-a-d*x-f*x^2) - exp(-a-b*x-c*x^2)
  grad = c(d1, d2, d3, d4, d5, d6)
  return(grad)
}

# Box-Cox Weibull model from Backhaus et al (2000)
# P(x) = 1 - exp( -exp(theta1 + theta2 * (x^theta3 - 1)/theta3) )
grad.boxcoxweibull = function(x, theta) {

  # identify parameters
  a = theta[1]
  b = theta[2]
  c = theta[3]

  # gradient components
  d1 = exp(-exp(a + b*(x^c-1)/c) + a + b*(x^c-1)/c)
  d2 = (x^c - 1)*exp(-exp(a + b*(x^c-1)/c) + a + b*(x^c-1)/c)/c
  d3 = (b*x^c * log(x)/c - b*(x^c-1)/c^2) * exp(-exp(a + b*(x^c-1)/c) + a + b*(x^c-1)/c)
  grad = c(d1, d2, d3)
  return(grad)

}

# Weibull model
# using version found in BMDS
# P[dose] = g + (1 - g) * (1 - exp(-b * dose^a))
grad.weibull = function(x, theta) {

  g = theta[1]
  a = theta[2]
  b = theta[3]

  g1 = exp(-b * x^a)
  g2 = -b * (g - 1) * x^a * log(x) * exp(-b * x^a)
  g3 = (g - 1) * x^a * (-exp(-b * x^a))
  return(c(g1, g2, g3))
}

# log-logistic model
# using version from BMDS
# P[dose] = g + (1 - g)/(1 + exp(-a - b * Log(dose)))
grad.loglogistic = function(x, theta) {

  g = theta[1]
  a = theta[2]
  b = theta[3]

  g1 = 1/(exp(a) * x^b + 1)
  g2 = (1-g)*exp(-a-b*log(x))/(exp(-a-b*log(x)) + 1)^2
  g3 = (1-g)*log(x)*exp(-a-b*log(x))/(exp(-a-b*log(x))+1)^2
  return(c(g1, g2, g3))
}

# D optimality
# maximize logdetM
obj.D = function(M, param) {
  suppressWarnings(log(det(M)))
}

# A optimality
# minimize trM^-1
obj.A = function(M, param) {

  # check if matrix is invertible
  if (!checkMinv(M))
    return(-Inf)
  else
    return(-sum(diag(solve(M))))
}

# BMD optimality
obj.bmd = function(M, param) {

  lambda = param[1]
  c = param[-1]
  if (!checkMinv(M))
    return(-Inf)
  else {
    Minv = solve(M)
    Dval = suppressWarnings(log(det(M)))
    Cval = -suppressWarnings(log(t(c) %*% Minv %*% c)) # note the sign flip
    p = length(c)
    return(lambda * Cval + (1 - lambda)/p * Dval)
  }
}

# derivatives of objective functions with respect to information matrix
# matrix singularity is already checked here
# M: information matrix
dPsi.D = function(M, param) {
  Minv = solve(M)
  return(Minv)
}

dPsi.A = function(M, param) {
  Minv = solve(M)
  Minv2 = Minv %*% Minv
  return(Minv2)
}

# compound D and c
# see Atkinson book p389
dPsi.CD = function(M, param) {

  Minv = solve(M)
  p = nrow(M)
  lambda = param[1]
  c = param[-1]
  num = Minv %*% c %*% t(c) %*% Minv
  denom = c(t(c) %*% Minv %*% c)

  return((1 - lambda)/p * Minv + lambda/denom * num)
}

################################################################################
# Main driver functions
################################################################################
find_design_single = function(
    grad_fun,
    obj,
    theta,
    bound,
    pts,
    algorithm,
    swarm,
    iter,
    seed
) {

  # design objective
  if (obj == "D")
    obj_fun = obj.D
  else if (obj == "A")
    obj_fun = obj.A
  else
    stop("Objective not supported")

  # objective function
  param = c()
  obj_fun_M = obj_fun_factory(grad_fun, obj_fun, theta, param)

  # set up variable bounds
  rangeVar = matrix(c(rep(c(0, bound), pts), rep(c(0,1), pts)), nrow = 2)

  # algorithm options
  control = list(numPopulation = swarm, maxIter = iter)

  # find design
  result = metaheuristicOpt::metaOpt(
    obj_fun_M,
    optimType = "MAX",
    algorithm = algorithm,
    numVar = 2 * pts,
    rangeVar,
    control,
    seed = seed
  )

  # check optimality
  vars = result$result
  x = vars[1:pts]
  w = vars[(pts+1):(2*pts)]
  M = M.nonlinear(x, w, theta, grad_fun)
  problem = list(bound = bound, obj = obj, theta = theta)
  p = plot_sens(x, w, problem, M, grad_fun)

  return(list(result = result, plot = p))
}

# function for computing design efficiencies
compute_eff = function(
    model,
    theta,
    objective,
    d1,
    d2,
    w1,
    w2
) {


  # select gradient function
  if (model == "Logistic")
    grad_fun = grad.logistic
  else if (model == "Logistic quadratic")
    grad_fun = grad.logistic.quad
  else if (model == "Logistic cubic")
    grad_fun = grad.logistic.cubic
  else if (model == "Logistic fractional polynomial")
    grad_fun = grad.logistic.fp
  else if (model == "Weibull")
    grad_fun = grad.weibull
  else if (model == "Log-logistic")
    grad_fun = grad.loglogistic
  else if (model == "Mixture multistage")
    grad_fun = grad.mix2
  else if (model == "Box-Cox Weibull")
    grad_fun = grad.boxcoxweibull

  if (objective == "D")
    obj_fun = obj.D
  else if (objective == "A")
    obj_fun = obj.A

  # define objective function
  param = c()
  obj_fun_M = obj_fun_factory(grad_fun, obj_fun, theta, param)

  # compute and return efficiencies
  if (objective == "D")
    (exp(obj_fun_M(c(d1, w1)))/exp(obj_fun_M(c(d2, w2))))^(1/length(theta))
  else if (objective == "A")
    obj_fun_M(c(d2, w2))/obj_fun_M(c(d1, w1))
}

# function for finding BMD designs
# basically a compound optimal design
# lambda is the weight parameter
find_bmd_design = function(
    model,
    lambda,
    risk,
    risk_type,
    theta,
    bound,
    pts,
    algorithm,
    swarm,
    iter,
    seed
) {

  # select gradient functions
  if (model == "Logistic") {
    grad_fun = grad.logistic

    if (risk_type == "Added") {
      bmd_grad = bmdgrad.logistic.add
    }
    else if (risk_type == "Extra")
      bmd_grad = bmdgrad.logistic.extra
  }
  else if (model == "Weibull") {
    grad_fun = grad.weibull

    if (risk_type == "Added") {
      bmd_grad = bmdgrad.weibull.add
    }
    else if (risk_type == "Extra") {
      bmd_grad = bmdgrad.weibull.extra
    }
  }
  else if (model == "Log-logistic") {
    grad_fun = grad.loglogistic

    if (risk_type == "Added") {
      bmd_grad = bmdgrad.loglogistic.add
    }
    else if (risk_type == "Extra") {
      bmd_grad = bmdgrad.loglogistic.extra
    }
  }


  # objective function
  c = bmd_grad(risk, theta) # compute BMD gradient components
  param = c(lambda, c)
  obj_fun = obj.bmd
  obj_fun_M = obj_fun_factory(grad_fun, obj_fun, theta, param)

  # set up variable bounds
  rangeVar = matrix(c(rep(c(0, bound), pts), rep(c(0,1), pts)), nrow = 2)

  # algorithm options
  control = list(numPopulation = swarm, maxIter = iter)

  # find design
  result = metaheuristicOpt::metaOpt(
    obj_fun_M,
    optimType = "MAX",
    algorithm = algorithm,
    numVar = 2 * pts,
    rangeVar,
    control,
    seed = seed
  )

  # check optimality
  vars = result$result
  x = vars[1:pts]
  w = vars[(pts+1):(2*pts)]
  M = M.nonlinear(x, w, theta, grad_fun)
  problem = list(bound = bound, obj = "bmd", theta = theta, param = param)
  p = plot_sens(x, w, problem, M, grad_fun)

  # compute D-efficiency of design
  #D_eff = compute_eff(model, theta, "bmd", )
  # probably need to make this an option

  return(list(result = result, plot = p))

}
################################################################################
# Core optimal design related functions
################################################################################
# function for constucting objective function for use with algorithms
# M_fun: information matrix function
# obj_fun: objective function of information matrix
# theta: parameter values to pass through to M_fun
# par: other parameters, such as c values for c objective
obj_fun_factory = function(grad_fun, obj_fun, theta, param) {

  # these are used in interface function
  force(grad_fun)
  force(theta)
  force(param)

  # interface called by optimization software
  # return this function
  function(vars, ...) {
    # distinguish between points and weights
    pts = length(vars)/2
    x = vars[1:pts]
    w = vars[(pts+1):(2*pts)]

    # check weight constraint
    s = sum(w, na.rm = T) # na.rm needed to fix if statement error
    if (s > 1) # constraint implementation
      return(-Inf)

    M_fun = M.nonlinear # always using general nonlinear matrix
    obj_value = obj_fun(M_fun(x, w, theta, grad_fun), param)

    # deal with missing
    if (is.na(obj_value))
      return(-Inf)
    else
      return(obj_value)
  }

}


# information matrix
# x: array of design points
# w: array of weights
# theta: array of parameter values
# most general case
# grad_fun: gradient function to use
M.nonlinear = function(x, w, theta, grad_fun) {

  IM = 0
  for (i in 1:length(x)) {
    IM_i = w[i] * grad_fun(x[i], theta) %*% t(grad_fun(x[i],theta))
    IM = IM + IM_i
  }
  IM
}


# plot sensitivity function for a given design
# problem is same list from toxODmeta
# x, w are design point and weight vectors
# M: pre-computed information matrix
# grad_fun: gradient function
plot_sens = function(x, w, problem, M, grad_fun) {

  # x values
  step = problem$bound/1000
  xvals = seq(0, problem$bound, step)

  # select derivative function for sensitivity function
  if (problem$obj == "D") {
    dPsi = dPsi.D
    param = NULL
  }
  else if (problem$obj == "A") {
    dPsi = dPsi.A
    param = NULL
  }
  else if (problem$obj == "bmd") {

    dPsi = dPsi.CD
    param = problem$param
  }
  else {
    # expand this to handle solving design problems with no verification
    #stop("No derivative specified for this objective.")
    # use y=2 to denote missing derivative function
    yvals = rep(2, length(xvals))
  }

  # compute sensitivity function
  # check first if matrix is invertible and then invert
  if (!checkMinv(M)) {
    # using y=1 to denote matrix singularity
    yvals = rep(1, length(xvals))
  }
  else {
    Minv = solve(M)
    yvals = sapply(xvals, sens, grad_fun, dPsi, M, problem$theta, param)
  }



  # plot
  # display message if missing matrix deriv or singular matrix
  if (sum(yvals - 1, na.rm = T) == 0) {
    p = ggplot2::ggplot(mapping = ggplot2::aes(y = yvals, x = xvals)) +
      ggplot2::geom_line(color = "blue") +
      ggplot2::geom_hline(yintercept = 0) +
      ggplot2::theme_bw() +
      ggplot2::labs(title = "Equivalence Theorem Check") +
      ggplot2::xlab("x") +
      ggplot2::ylab("ch(x)") +
      ggplot2::annotate("text", x = mean(xvals), y = 0.5,
                        label = "Singular information matrix", size = 5)
  }
  else if (sum(yvals - 2, na.rm = T) == 0) {
    p = ggplot2::ggplot(mapping = ggplot2::aes(y = yvals, x = xvals)) +
      ggplot2::geom_line(color = "blue") +
      ggplot2::geom_hline(yintercept = 0) +
      ggplot2::theme_bw() +
      ggplot2::labs(title = "Equivalence Theorem Check") +
      ggplot2::xlab("x") +
      ggplot2::ylab("ch(x)") +
      ggplot2::annotate("text", x = mean(xvals), y = 0.5,
                        label = "No dPsi defined", size = 5)
  }
  else {
    p = ggplot2::ggplot(mapping = ggplot2::aes(y = yvals, x = xvals)) +
      ggplot2::geom_line(color = "blue") +
      ggplot2::geom_hline(yintercept = 0) +
      #ggplot2::geom_point(aes(x = design_points, y = pts_ch), col = "red", size = 3) +
      ggplot2::geom_vline(xintercept = x, color = "red", linetype = "dashed") +
      ggplot2::theme_bw() +
      ggplot2::labs(title = "Equivalence Theorem Check") +
      ggplot2::xlab("x") +
      ggplot2::ylab("ch(x)")
  }


  return(p)
}

# sensitivity function

# general function
# z: independent variable
# grad: gradient function
# dPsi: derivative of the objective function wrt M
# M: information matrix
# theta: model parameters
sens = function(z, grad, dPsi, M, theta, param) {

  dg = grad(z, theta)
  dM = dPsi(M, param)
  y = t(dg) %*% dM %*% dg - sum(diag(M %*% dM))
  return(y)
}

################################################################################
# Other utility functions
################################################################################
# function that displays latex formulas for models in app
model_display = function(model) {

  if (model == "Dichotomous Hill")
    "$$ P(d) = \\theta_1 + \\frac{(\\theta_2 - \\theta_2 \\theta_1)}{1 + \\exp(-\\theta_3 - \\theta_4\\log (d))} $$"
  # else if (model == "Gamma") # don't know how to do this => Elvis' paper reparameterizes
  #   "$$ P(d) = $$"
  else if (model == "Logistic")
    "$$ P(d) = \\frac{1}{1 + \\exp(-\\theta_1 - \\theta_2 d)} $$"
  else if (model == "Logistic quadratic")
    "$$P(d) = \\frac{1}{1 + \\exp(-\\theta_1 - \\theta_2 d - \\theta_3 d^2)}$$"
  else if (model == "Logistic cubic")
    "$$P(d) = \\frac{1}{1 + \\exp(-\\theta_1 - \\theta_2 d - \\theta_3 d^2 - \\theta_4 d^3)}$$"
  else if (model == "Logistic fractional polynomial")
    "$$P(d) = \\frac{1}{1 + \\exp(-\\theta_1 - \\theta_2 d^{\\theta_4} - \\theta_3 d^{\\theta_5})}$$"
  else if (model == "Log-logistic")
    "$$ P(d) = \\theta_1 +  \\frac{1-\\theta_1}{1 + \\exp(-\\theta_2- \\theta_3 \\log d)}$$"
  else if (model == "Log-probit")
    "$$ P(d) = \\theta_1 + (1 - \\theta_1) \\Phi(\\theta_2 + \\theta_3 \\log(d))$$"
  else if (model == "Multistage degree 1")
    "$$P(d) = \\theta_1 + (1 - \\theta_1)(1 - \\exp(-\\theta_2 d))$$"
  else if (model == "Multistage degree 2")
    "$$P(d) = \\theta_1 + (1 - \\theta_1)(1 - \\exp(-\\theta_2 d - \\theta_3 d^2))$$"
  else if (model == "Multistage degree 3")
    "$$P(d) = \\theta_1 + (1 - \\theta_1)(1 - \\exp(-\\theta_2 d - \\theta_3 d^2 - \\theta_4 d^3))$$"
  else if (model == "Probit")
    "$$P(d)=\\Phi(\\theta_1 + \\theta_2 d)$$"
  else if (model == "Quantal linear")
    "$$P(d) = \\theta_1 + (1-\\theta_1)(1-\\exp(-\\theta_2 d))$$"
  else if (model == "Weibull")
    "$$P(d) = \\theta_1 + (1-\\theta_1)(1-\\exp(-\\theta_2 d^{\\theta_3}))$$"
  else if (model == "Mixture multistage")
    "$$P(d) = \\theta_6 \\left[1 - \\exp(-\\theta_1-\\theta_2 d - \\theta_3 d^2) \\right] + (1-\\theta_6)\\left[1 - \\exp(-\\theta_1 - \\theta_4 d - \\theta_5 d^2) \\right]$$"
  else if (model == "Box-Cox Weibull")
    "$$P(d)=1-\\exp \\left[ -\\exp \\left(\\theta_1 + \\theta_2 \\frac{d^{\\theta_3}-1}{\\theta_3}\\right)\\right]$$"
  else
    "Model not supported"

}

# convert raw text input to a vector of parameter values
# pulling this out into its own function because input checking could be complex
# useful in multiple places where there is text input
process_theta = function(text) {
  as.numeric(strsplit(text, ",")[[1]])
}

# checks if information matrix is invertible
# returns 1 if invertible and 0 if not
# can optimize easily for 2 dim
checkMinv = function(M) {

  if (class(try(solve(M),silent=T))[1]!="matrix")
    return(0)
  else
    return(1)
}

# utility functions for fractional polynomials

# Box-Tidwell transformation
bt = function(X, p) {
  if (p != 0)
    return(X^p)
  else if (p == 0)
    return(suppressWarnings(log(X)))
}

# derivative of Box-Tidwell
dbt = function(X, p) {
  if (p != 0)
    return(p * X^(p-1))
  else if (p == 0)
    return(1/X)
}

# H function
# j: index
H = function(j, X, powers) {
  if (j == 1) # base case
    return(1)
  if (powers[j] != powers[j-1])
    return(bt(X, powers[j]))
  else if (powers[j] == powers[j-1])
    return(suppressWarnings(log(X)) * H(j-1, X, powers))
}

# derivative of the H function
dH = function(j, X, powers) {
  if (j == 1) # base case
    return(0)
  if (powers[j] != powers[j-1])
    return(dbt(X, powers[j]))
  else if (powers[j] == powers[j-1])
    return(suppressWarnings(log(X)) * dH(j-1, X, powers) +
             H(j-1, X, powers)/X)
}

# calculates the fractional polynomial for given X, coefficients, powers
# m: degree
fracpoly = function(X, betas, powers, m) {

  y = 0

  for (j in 1:(m+1)) {
    y = y + betas[j] * H(j, X, powers)
  }

  return(y)

}

################################################################################
# gradient functions for BMD
################################################################################
# refer to the EPA's BMDS user manual for the original formulas
bmdgrad.logistic.add = function(r, theta) {

  beta0 = theta[1]
  beta1 = theta[2]

  g1 = r/(beta1 * (exp(beta0) + r))
  g2 = -log(-(exp(beta0) * (r - 1))/(exp(beta0) +r)) / beta1^2
  return(c(g1, g2))
}

bmdgrad.logistic.extra = function(r, theta) {

  beta0 = theta[1]
  beta1 = theta[2]

  g1 = (exp(beta0) + 1)*r*(exp(beta0)*(r+1)+r-1)/(beta1*(exp(beta0)*r+r-1)*(exp(beta0)*(r+1)+r))
  g2 = - log(- (exp(beta0)*(exp(beta0)*r+r-1))/(exp(beta0)*(r+1)+r))/beta1^2
  return(c(g1, g2))

}

bmdgrad.weibull.add = function(r, theta) {

  g = theta[1]
  a = theta[2]
  b = theta[3]

  g1 = r * (-log((g+r-1)/(g-1))/b)^(1/(a-1)) / (a*b*(g-1)*(g+r-1))
  g2 = -(log(-(log((g+r-1)/(g-1)))/b) * (-(log((g+r-1)/(g-1)))/b)^(1/a))/(a^2)
  g3 = -(-log((g+r-1)/(g-1))/b)^(1/a)/(a * b)
  return(c(g1, g2, g3))
}

bmdgrad.weibull.extra = function(r, theta) {

  g = theta[1]
  a = theta[2]
  b = theta[3]

  g1 = 0
  g2 = - log(-log(1-r)/b)*(-log(1-r)/b)^(1/a) / a^2
  g3 = - (-log(1-r)/b)^(1/a) / (a*b)
  return(c(g1, g2, g3))
}

bmdgrad.loglogistic.add = function(r, theta) {

  g = theta[1]
  a = theta[2]
  b = theta[3]

  g1 = exp(-a/b)*(-r/(g+r-1))^(1/(b+1)) / (b*r)
  g2 = -exp(-a/b)*(-r/(g+r-1))^(1/b) / b
  g3 = exp(-a/b)*(-r/(g+r-1))^(1/b) * (a - log(-r/(g+r-1))) / b^2
  return(c(g1, g2, g3))
}

bmdgrad.loglogistic.extra = function(r, theta) {

  g = theta[1]
  a = theta[2]
  b = theta[3]

  g1 = 0
  g2 = -exp((log(r/(1-r))-a)/b)/b
  g3 = exp(-a/b) * (r/(1-r))^(1/b) * (a - log(r/(1-r))) / b^2
  return(c(g1,g2,g3))
}



################################################################################
# Run the application
################################################################################
shinyApp(ui = ui, server = server)
