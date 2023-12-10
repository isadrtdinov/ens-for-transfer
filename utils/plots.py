import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


def draw_simplex(alphas, values, title='', num_pixels=1000, size=12, bandwidth_factor=0.01):
    values = np.array(values)
    ratio = np.sqrt(3) / 2
    plt.clf()
    plt.figure(figsize=(size, size * ratio))
    plt.rcParams.update({'font.size': 14})

    # project alphas on a triangle
    points = np.array([[0, 0], [1, 0], [0.5, ratio]])
    alpha_points = alphas @ points

    # interpolate values inside triangle
    bandwidth = (np.floor(np.sqrt(alphas.shape[0] * 2)) - 1)  # set bandwidth equal to grid step
    bandwidth *= bandwidth_factor
    knn = KNeighborsRegressor(n_neighbors=alphas.shape[0], weights=lambda dist: np.exp(-np.square(dist) / bandwidth))
    knn.fit(alpha_points, values)

    # create values grid
    x_pixels, y_pixels = num_pixels, int(num_pixels * ratio)
    x = np.linspace(0, 1, x_pixels)
    y = np.linspace(0, ratio, y_pixels)
    x, y = np.meshgrid(x, y)
    x, y = x.ravel(), y.ravel()
    grid_values = knn.predict(np.stack([x, y], axis=1))

    # cut off values outside triangle
    grid_values[y >= np.sqrt(3) * x] = np.nan
    grid_values[y >= -np.sqrt(3) * (x - 1)] = np.nan
    grid_values = grid_values.reshape(y_pixels, x_pixels)

    # plot grid values using imshow and real values using scatter plot
    plt.imshow(grid_values, origin='lower', cmap='cool', vmin=values.min(), vmax=values.max())
    plt.scatter(alpha_points[:, 0] * x_pixels, alpha_points[:, 1] * x_pixels,
                c=values,  vmin=values.min(), vmax=values.max(),
                edgecolors='black', s=100, linewidth=2, cmap='cool')
    plt.colorbar()

    # add margin to limits
    margin = int(x_pixels * 0.03)
    plt.xlim(-margin, x_pixels + margin)
    plt.ylim(-margin, y_pixels + margin)

    # set x and y ticks
    x_labels = np.arange(0, 1.01, 0.2)
    x_ticks = (x_labels * x_pixels).astype(int)
    x_labels = [f'{val:.1f}' for val in x_labels]
    plt.xticks(ticks=x_ticks, labels=x_labels)

    y_labels = np.arange(0, np.sqrt(3) / 2, 0.2)
    y_ticks = (y_labels * x_pixels).astype(int)
    y_labels = [f'{val:.1f}' for val in y_labels]
    plt.yticks(ticks=y_ticks, labels=y_labels)

    plt.title(title)
    return plt
