from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
import os
import time

def ackley(x):
    '''Ackley function'''
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) - \
        np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + \
        np.exp(1) + 20

def rosenbrock(x):
    '''Rosenbrock function'''
    return 100 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1) ** 2

def PSO(function, bounds, swarm_size=10, inertia=0.5, pa=0.8, ga=0.9,
        max_vnorm=10, num_iters=100, verbose=False, func_name=None):
    '''Particle Swarm Optimization (PSO)
    Params:
        - function      : function to be optimized
        - bounds        : list, bounds of each dimension
        - swarm_size    : int, the population size of the swarm
        - intertia      : float, coefficient of momentum
        - pa            : float, personal acceleration
        - ga            : float, global acceleration
        - max_vnorm     : max velocity norm
        - num_iters     : int, number of iterations
        - verbose       : boolean, whether to print results or not
        - func_name     : the name of object function to optimize
    
    Returns:
        - history       : history of particles and global bests
    '''

    bounds = np.array(bounds)
    assert np.all(bounds[:, 0] < bounds[:, 1])
    
    dimensions = len(bounds)
    X = np.random.rand(swarm_size, dimensions)
    print('### Optimize:', func_name)

    def clip_by_norm(x, max_norm):
        norm = np.linalg.norm(x)
        return x if norm <= max_norm else x * max_norm / norm
    

    # Step 1: Initialize all particle randomly in the search-space
    particles = X * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    velocities = X * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    personal_bests = np.copy(particles)
    personal_best_fitness = [np.inf for _ in particles]

    global_best_index = np.argmin(personal_best_fitness)
    global_best = personal_bests[global_best_index]
    global_best_fitness = function(global_best)

    history = {
        'particles': [],
        'global_best_fitness': [],
        'global_best': [[np.inf, np.inf] for _ in range(num_iters)],
        'obj_func':func_name
    }

    
    # Step 2: Iteration starts
    for i in range(num_iters):
        history['particles'].append(particles)
        history['global_best_fitness'].append(global_best_fitness)
        history['global_best'][i][0] = global_best[0]
        history['global_best'][i][1] = global_best[1]

        if verbose:
            print(f'iteration# {i}:', end=' ')

        # Step 3: Evaluate current swarm
        # personal best
        for p_i in range(swarm_size):
            fitness = function(particles[p_i])
            if fitness < personal_best_fitness[p_i]:
                personal_bests[p_i] = particles[p_i]
                personal_best_fitness[p_i] = fitness

        # global best
        if np.min(personal_best_fitness) < global_best_fitness:
            global_best_index = np.argmin(personal_best_fitness)
            global_best = personal_bests[global_best_index]
            global_best_fitness = function(global_best)

        # Step 4: Calculate the acceleration and momentum
        momentum = inertia * velocities
        local_acc = pa * np.random.rand() * (personal_bests - particles)
        global_acc = ga * np.random.rand() * (global_best - particles)

        # Step 5: Update the velocities
        velocities = momentum + local_acc + global_acc
        velocities = clip_by_norm(velocities, max_vnorm)

        # Step 6: Update the position of particles
        particles += velocities

        # logging
        if verbose:
            print(f'Fitness: {global_best_fitness:.5f}, \
                  Position: {global_best}, Velocity: {np.linalg.norm(velocities)}')
    
    return history


def visualize_history_2D(function=None, history=None, bounds=None, 
                         minima=None, func_name='', save2mp4=False, save2gif=False):
    '''Visualize the process of optimizing
    Params:
        - function  : object function
        - history   : dict, object returned from PSO()
        - bounds    : list, bounds of each dimension
        - minima    : list, the exact minima to show in the plot
        - func_name : str, the name of the object function
        - save2mp4  : bool, whether to save as mp4 or not
        - save2gif  : bool, whether to save as GIF or not
    '''

    print(f'Visualizing optimization {func_name}')
    assert len(bounds) == 2

    x = np.linspace(bounds[0][0], bounds[0][1], 50)
    y = np.linspace(bounds[1][0], bounds[1][1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.array([function([x, y]) for x, y in zip(X, Y)])

    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121, facecolor='w')
    ax2 = fig.add_subplot(122, facecolor='w')

    def animate(frame, history):
        ax1.cla()
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_title(f'{func_name} | iter={frame+1} | \
                      Gbest=({history["global_best"][frame][0]}, {history["global_best"][frame][1]})')
        ax1.set_xlim(bounds[0][0], bounds[0][1])
        ax1.set_ylim(bounds[1][0], bounds[1][1])

        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness')
        ax2.set_title(f'Minima Value Plot | Population={len(history["particles"][0])} \
                      | MinValue={history["global_best_fitness"][frame]}')
        ax2.set_xlim(2, len(history['global_best_fitness']))
        ax2.set_ylim(10e-16, 10e0)
        ax2.set_yscale('log')

        data = history['particles'][frame]
        global_best = np.array(history['global_best_fitness'])

        contour = ax1.contour(X, Y, Z, levels=50, cmap='magma')
        ax1.plot(minima[0], minima[1], marker='o', color='black')

        ax1.scatter(data[:, 0], data[:, 1], marker='x', color='black')

        if frame > 1:
            for i in range(len(data)):
                ax1.plot(
                    [history['particles'][frame-n][i][0] for n in range(2, -1, -1)],
                    [history['particles'][frame-n][i][1] for n in range(2, -1, -1)]
                )
        elif frame == 1:
            for i in range(len(data)):
                ax1.plot(
                    [history['particles'][frame-n][i][0] for n in range(1, -1, -1)],
                    [history['particles'][frame-n][i][1] for n in range(1, -1, -1)]
                )
        
        x_range = np.arange(1, frame+2)
        ax2.plot(x_range, global_best[:frame+1])
    
    fig.suptitle(f'Optimization of {func_name.split()[0]} function by PSO, \
                 f_min({minima[0]}, {minima[1]})={function(minima)}', fontsize=20)
    
    ani = animation.FuncAnimation(fig, animate, fargs=(history,),
                                  frames=len(history['particles']), interval=250, repeat=False, blit=False)
    
    if save2mp4:
        os.makedirs('mp4/', exist_ok=True)
        ani.save(f'mp4/PSO_{func_name.split()[0]}_population_{len(history["particles"][0])}.mp4', 
                 writer='ffmpeg', dpi=100)
    
    elif save2gif:
        os.makedirs('gif/', exist_ok=True)
        ani.save(f'gif/PSO_{func_name.split()[0]}_population_{len(history["particles"][0])}.gif', 
                 writer='ffmpeg', dpi=100)
    
    else:
        plt.show()

def experiment_suits():
    '''Perform PSO experiments
    Current test set: ["Rosenbrock function", "Ackley function"]
    '''

    save2mp4, save2gif = False, True
    object_functions = [rosenbrock, ackley]
    object_functions_names = ['Rosenbrock Function', 'Ackley Function']

    each_boundaries = [
        [[-2, 2], [-2, 2]],
        [[-32, 32], [-32, 32]]
    ]

    global_minima = [[1, 1], [0, 0]]

    swarm_sizes = [5, 15, 30, 35, 100]
    num_iterations = 100

    for function, function_name, bounds, global_minimum in zip(
        object_functions, object_functions_names, each_boundaries, global_minima):
        for swarm_size in swarm_sizes:
            history = PSO(
                function=function,
                bounds=bounds,
                swarm_size=swarm_size,
                num_iters=num_iterations,
                verbose=False,
                func_name=function_name,
            )

            print(f'Global best: {history["global_best_fitness"][-1]}, \
                  Global best position: {history["global_best"][-1]}')
            
            visualize_history_2D(
                function=function,
                history=history,
                bounds=bounds,
                minima=global_minimum,
                func_name=function_name,
                save2mp4=save2mp4,
                save2gif=save2gif
            )


if __name__ == '__main__':
    start = time.time()
    experiment_suits()
    end = time.time()
    print(f'The experiment took {end-start} units')
