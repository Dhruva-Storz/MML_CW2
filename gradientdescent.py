from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np



def f1(x):
    B = np.array([[4,-2],[-2,4]])
    a = np.array([0, 1])
    b = np.array([-2, 1])
    f = x.T @ B @ x - x.T @ x + a.T @ x - b.T @ x
    return f


def f2(x):
    B = np.array([[4,-2],[-2,4]])
    a = np.array([0,1])
    b = np.array([-2,1])
    xa = x-a
    xb = x-b

    f = np.cos(xb.T @ xb) + xa.T @ B @ xa
    return f

def f3(x):
    B = np.array([[4,-2],[-2,4]])
    a = np.array([0, 1])
    b = np.array([-2, 1])
    xa = x - a
    xb = x - b
    f = 1 - (np.exp(-1*(xa.T @ xa)) + np.exp(-1*(xb.T @ B @ xb)) - 0.1*np.log(np.linalg.det(0.01*np.identity(2)+np.outer(x,x.T))))
    return f


def grad_f1(x):
    B = np.array([[4,-2],[-2,4]])
    a = np.array([0, 1])
    b = np.array([-2, 1])
    f = 2*(x.T @ B) - 2*x.T + (a-b)
    return f


def grad_f2(x):
    B = np.array([[4,-2],[-2,4]])
    a = np.array([0, 1])
    b = np.array([-2, 1])
    xa = x - a
    xb = x - b
    f = -2 * xb.T * np.sin(xb.T @ xb) + 2 * xa.T @ B
    return f

def grad_f3(x):
    B = np.array([[4,-2],[-2,4]])
    a = np.array([0, 1])
    b = np.array([-2, 1])
    xa = x - a
    xb = x - b
    ddx_xxT = np.array([
            [[2*x[0] , x[1]],[x[1],0]] ,
            [[ 0 , x[0] ],[ x[0] , 2*x[1]]]
            ])

    # keep getting weird singular matrix error when the inverse function is multiplied by 1 or when gamma is > 0.1.
    # so i had to isolate the third part (p3) of the gradient equation
    try:
        p3 = (np.matrix.trace(np.linalg.inv(0.01 * np.identity(2) + np.outer(x, x.T)) @ ddx_xxT))
    except np.linalg.LinAlgError:
        p3 = 0
        print(f'x = {x[0],x[1]}')


    f = +2 * xa.T * np.exp(-1 * xa.T @ xa) \
        +2 * xb.T @ B * np.exp(-1 * xb.T @ B @ xb) \
        +0.1*p3

    return f

def plot3D(function):
    import matplotlib.pyplot as plt
    x1 = np.linspace(-10,10,100)
    x2 = np.linspace(-10,10,100)
    Z = np.zeros((100,100))
    for index1,val1 in enumerate(x1):
        for index2,val2 in enumerate(x2):
            Z[index2,index1] = function(np.array([val1,val2]))

    X,Y = np.meshgrid(x1,x2)
    #
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    #ax.plot_wireframe(X, Y, Z, color='green')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')

    #plt.contour(X, Y, Z, 10, cmap='winter');
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                     cmap='cool', edgecolor='none')
    ax.set_title('surface');
    #
    plt.show()

def plot3Dgrad(function,index):
    x1 = np.linspace(-10,10,100)
    x2 = np.linspace(-10,10,100)
    Z = np.zeros((100,100))
    for index1,val1 in enumerate(x1):
        for index2,val2 in enumerate(x2):
            Z[index2,index1] = function(np.array([val1,val2]))[index]

    X,Y = np.meshgrid(x1,x2)
    #
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    #ax.plot_wireframe(X, Y, Z, color='green')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')

    #plt.contour(X, Y, Z, 10, cmap='winter');
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                     cmap='cool', edgecolor='none')
    ax.set_title('surface');
    #
    plt.show()




def gradient_descent(function,gradient):
    #sets labels for plotting and printing purposes
    if function == f1:
        label = "f1(x)"
    elif function == f2:
        label = "f2(x)"
    elif function == f3:
        label = "f3(x)"

    # gradient descent parameter intialization
    x1_init = 0.3
    x2_init = 0
    x1_new = x1_init
    x2_new = x2_init
    x1_old = x1_new
    x2_old = x2_new
    gamma = 0.01
    error_tolerance = 10e-4
    count = 0
    #gradient descent parameter intialization

    # plotting
    plotrange = 1
    x1 = np.linspace(-plotrange, plotrange, 100)
    x2 = np.linspace(-plotrange+0.5, plotrange+0.5, 100)
    Z = np.zeros((100, 100))
    for index1, val1 in enumerate(x1):
        for index2, val2 in enumerate(x2):
            Z[index2, index1] = function(np.array([val1, val2]))
    X, Y = np.meshgrid(x1, x2)
    ax = plt.subplot()
    ax.contour(X, Y, Z, 100, cmap='winter');
    plt.title(f"Contour Plot of Gradient Descent: {label}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    #plotting

    #gradient descent loop
    while (count < 50) and ((abs(function(np.array([x1_new,x2_new])-function(np.array([x1_old,x2_old])))) > error_tolerance) \
            or (abs(x1_new-x1_old)+abs(x2_new-x2_old) > 2*error_tolerance)): # and abs(x2_new-x2_old) > error_tolerance)):
        x1_old = x1_new
        x2_old = x2_new
        x1_new = x1_old - gamma * gradient(np.array([x1_old, x2_old]))[0]
        x2_new = x2_old - gamma * gradient(np.array([x1_old, x2_old]))[1]
        print(x1_new)
        print(x2_new)
        count+=1
        ax.annotate('', xy=(x1_new,x2_new), xytext=(x1_old,x2_old),
                     arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                     va='center', ha='center')
    #gradient descent loop

    #final outputs
    print(f'{label} minimum after {count} iterations at : {[x1_new, x2_new]}. At minimum, {label} = {function(np.array([x1_new,x2_new]))}')
    plt.savefig(f'{label}_GD.pdf')
    plt.show()




def main():
    # plot3D(f1)
    # plot3D(f2)
    #plot3D(f3)

    #plot3Dgrad(grad_f1,0)
    #plot3Dgrad(grad_f1, 1)
    #plot3Dgrad(grad_f2,0)
    # plot3Dgrad(grad_f2,1)
    #plot3Dgrad(grad_f3,0)
    #plot3Dgrad(grad_f3, 1)

    #gradient_descent(f1,grad_f1)
    gradient_descent(f2,grad_f2)
    #gradient_descent(f3,grad_f3)

    #print(grad_f3(np.array([3,2])))



if __name__ == '__main__':

    main()