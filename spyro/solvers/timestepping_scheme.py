from firedrake import dx
from ..sources import MMS_time, timedependentSource, sourceDerivative_in_source

def ssprk_timestepping_with_source(time_order, solv, b1, b2, dUP, UP0, UP, dt, K, model, t):
    if time_order == 3:
        return ssprk3_with_source(solv, b1, b2, dUP, UP0, UP, dt, K, model, t)
    elif time_order == 4:
        return ssprk4_with_source(solv, b1, b2, dUP, UP0, UP, dt, K, model, t)
    else:
        raise ValueError("This time order not yet implemented.")

def ssprk3_with_source(solv, b1, b2, dUP, UP0, UP, dt, K, model, t):

    solv.solve(dUP, b)  # Solve for du and dp
    K.assign(dUP)

    # Second step
    UP.assign(UP0 + dt * K)

    # solv.solve() #Solve for du and dp
    solv.solve(dUP, b)  # Solve for du and dp
    K.assign(dUP)

    # Third step
    UP.assign(0.75 * UP0 + 0.25 * (UP + dt * K))

    # solve.solve() #Solve for du and dp
    solv.solve(dUP, b)  # Solve for du and dp
    K.assign(dUP)

    # Updating answer
    UP.assign((1.0 / 3.0) * UP0 + (2.0 / 3.0) * (UP + dt * K))

    
    return UP

def ssprk4_with_source(solv, b1, b2, dUP, UP0, UP, dt, K, model, t):
    freq = model["acquisition"]["frequency"]
    source            = timedependentSource(model, t, freq)
    source0 = source

    i=1
    while i < 6:
        solv.solve(dUP, b1)  # Solve for du and dp
        K.assign(dUP)
        solv.solve(dUP, b2)
        K.assign(K+dUP*source)
        source_derivative = 
        
        source = source + dt*source_derivative/6.
        UP.assign(UP + dt*K/6.)
        i+=1
    
    UP0.assign( (1./25.)*UP0 + (9./25.)*UP )
    UP.assign( 15*UP0 - 5*UP )

    while i<10:
        solv.solve(dUP, b1)  # Solve for du and dp
        K.assign(dUP)
        solv.solve(dUP, b2)
        K.assign(K+dUP*source)

        source = source + dt*source_derivative/6.
        UP.assign( UP + dt*K/6. )
        i+=1

    solv.solve(dUP, b1)  # Solve for du and dp
    K.assign(dUP)
    solv.solve(dUP, b2)
    K.assign(K+dUP)
    UP.assign( UP0 + 3./5.*UP + (1./10.)*dt*K )

    return UP

def ssprk_timestepping_no_source(time_order, solv, b1, dUP, UP0, UP, dt, K):
    if time_order == 3:
        return ssprk3_with_source(solv, b1, dUP, UP0, UP, dt, K)
    elif time_order == 4:
        return ssprk4_with_source(solv, b1, dUP, UP0, UP, dt, K)
    else:
        raise ValueError("This time order not yet implemented.")

def ssprk3_no_source(solv, b1, dUP, UP0, UP, dt, K):

    solv.solve(dUP, b)  # Solve for du and dp
    K.assign(dUP)

    # Second step
    UP.assign(UP0 + dt * K)

    # solv.solve() #Solve for du and dp
    solv.solve(dUP, b)  # Solve for du and dp
    K.assign(dUP)

    # Third step
    UP.assign(0.75 * UP0 + 0.25 * (UP + dt * K))

    # solve.solve() #Solve for du and dp
    solv.solve(dUP, b)  # Solve for du and dp
    K.assign(dUP)

    # Updating answer
    UP.assign((1.0 / 3.0) * UP0 + (2.0 / 3.0) * (UP + dt * K))

    
    return UP

def ssprk4_no_source(solv, b1, dUP, UP0, UP, dt, K):
    
    i=1
    while i < 6:
        solv.solve(dUP, b1)  # Solve for du and dp
        K.assign(dUP)
        UP.assign(UP + dt*K/6.)
        i+=1
    
    UP0.assign( (1./25.)*UP0 + (9./25.)*UP )
    UP.assign( 15*UP0 - 5*UP )

    while i<10:
        solv.solve(dUP, b1)  # Solve for du and dp
        K.assign(dUP)
        UP.assign( UP + dt*K/6. )
        i+=1

    solv.solve(dUP, b1)  # Solve for du and dp
    K.assign(dUP)
    UP.assign( UP0 + 3./5.*UP + (1./10.)*dt*K )

    return UP