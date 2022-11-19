# Python3 implementation of the approach

# Function that returns true if
# the given pixel is valid
def isValid(screen, m, n, x, y, prevC, newC, tolerance):
    if x < 0 or x >= m \
            or y < 0 or y >= n or \
            not(screen[x][y] < (prevC + tolerance) and screen[x][y] > (prevC - tolerance)) \
            or screen[x][y] == newC:
        return False
    return True


# FloodFill function
def floodFill(screen,
              m, n, x,
              y, prevC, newC, tolerance, counter_limit):
    queue = []
    screen_cpy = screen.copy()
    counter = 0
    # Append the position of starting
    # pixel of the component
    queue.append([x, y])

    # Color the pixel with the new color
    screen_cpy[x][y] = newC

    # While the queue is not empty i.e. the
    # whole component having prevC color
    # is not colored with newC color
    while queue:

        # Dequeue the front node
        currPixel = queue.pop()

        posX = currPixel[0]
        posY = currPixel[1]

        # Check if the adjacent
        # pixels are valid
        if isValid(screen_cpy, m, n,
                   posX + 1, posY,
                   prevC, newC, tolerance):
            # Color with newC
            # if valid and enqueue
            screen_cpy[posX + 1][posY] = newC
            queue.append([posX + 1, posY])

        if isValid(screen_cpy, m, n,
                   posX - 1, posY,
                   prevC, newC, tolerance):
            screen_cpy[posX - 1][posY] = newC
            queue.append([posX - 1, posY])

        if isValid(screen_cpy, m, n,
                   posX, posY + 1,
                   prevC, newC, tolerance):
            screen_cpy[posX][posY + 1] = newC
            queue.append([posX, posY + 1])

        if isValid(screen_cpy, m, n,
                   posX, posY - 1,
                   prevC, newC, tolerance):
            screen_cpy[posX][posY - 1] = newC
            queue.append([posX, posY - 1])
        counter += 1

        if counter > counter_limit:
            return screen
    return screen_cpy

