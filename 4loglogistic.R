# estimating parameters for the 4 parameter log-logistic model
library(drc)

dat = data.frame(
  dose = c(0, 1.97, 8.27, 29.5),
  n = c(50, 50, 50, 50),
  incidence = c(1, 11, 21, 36)
)

mod = drm(incidence ~ dose, data = dat, fct = LL2.4())
summary(mod)
coef(mod)

# convert to parameterization used in the app
b = coef(mod)[[1]]
c = coef(mod)[[2]]
d = coef(mod)[[3]]
e = coef(mod)[[4]]

theta1 = c
theta2 = b*e
theta3 = -b
theta4 = d
c(theta1, theta2, theta3, theta4)
