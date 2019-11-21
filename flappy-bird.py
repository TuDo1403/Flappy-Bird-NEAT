import pygame
import neat
import time
import os
import random

pygame.font.init()

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800

BIRD_IMAGES = [pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird1.png"))),
               pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird2.png"))),
               pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird3.png")))]
PIPE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "pipe.png")))
BASE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "base.png")))
BG_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bg.png")))
STAT_FONT = pygame.font.SysFont("comicsans", 50)

class Bird:
    IMAGES = BIRD_IMAGES
    MAX_ROTATION = 25
    ROTATION_VELOCITY = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.image_count = 0
        self.image = self.IMAGES[0]

    def jump(self):
        self.velocity = -10.5   # for bird to move upward (positive velocity mean bird will go downward)
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        distance = self.velocity*self.tick_count + 1.5*self.tick_count**2
        if distance >= 16:  # if moving down too fast, no accelerate anymore
            distance = 16
        if distance < 0:    # if moving up, accelerate a little bit        
            distance -= 2
        
        self.y = self.y + distance

        if distance < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROTATION_VELOCITY
        
    def draw(self, window):
        self.image_count += 1

        if self.image_count < self.ANIMATION_TIME:
            self.image = self.IMAGES[0]
        elif self.image_count < self.ANIMATION_TIME*2:
            self.image = self.IMAGES[1]
        elif self.image_count < self.ANIMATION_TIME*3:
            self.image = self.IMAGES[2]
        elif self.image_count < self.ANIMATION_TIME*4:
            self.image = self.IMAGES[1]
        elif self.image_count < self.ANIMATION_TIME*4 + 1:
            self.image = self.IMAGES[0]
            self.image_count = 0
        
        if self.tilt <= -80:
            self.image = self.IMAGES[1]
            self.image_count = self.ANIMATION_TIME*2
        
        rotated_image = pygame.transform.rotate(self.image, self.tilt)
        new_rectangle = rotated_image.get_rect(center = self.image.get_rect(topleft = (self.x, self.y)).center)
        window.blit(rotated_image, new_rectangle.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.image)

class Pipe:
    GAP = 200
    VELOCITY = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 100

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMAGE, False, True)
        self.PIPE_BOTTOM = PIPE_IMAGE

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VELOCITY
    
    def draw(self, window):
        window.blit(self.PIPE_TOP, (self.x, self.top))
        window.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        bottom_point = bird_mask.overlap(bottom_mask, bottom_offset)
        top_point = bird_mask.overlap(top_mask, top_offset)

        if top_point or bottom_point:
            return True
        return False


class Base:
    VELOCITY = 5
    WIDTH = BASE_IMAGE.get_width()
    IMAGE = BASE_IMAGE

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VELOCITY
        self.x2 -= self.VELOCITY

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
    
    def draw(self, window):
        window.blit(self.IMAGE, (self.x1, self.y))
        window.blit(self.IMAGE, (self.x2, self.y))


def draw_window(window, bird, pipes, base, score):
    window.blit(BG_IMAGE, (0, 0))

    for pipe in pipes:
        pipe.draw(window)

    text = STAT_FONT.render("Score:" + str(score), 1,(255, 255, 255))
    window.blit(text, (WINDOW_WIDTH - 10 - text.get_width(), 10))
    
    base.draw(window)

    bird.draw(window)
    pygame.display.update()

def main():
    bird = Bird(230, 350)
    base = Base(730)
    pipes = [Pipe(700)]
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    score = 0

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        #bird.move()
        add_pipe = False
        removed_pipe = []
        for pipe in pipes:
            if pipe.collide(bird):
                pass

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                removed_pipe.append(pipe)
            
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True
            pipe.move()
        
        if add_pipe:
            score += 1
            pipes.append(Pipe(600))

        for item in removed_pipe:
            pipes.remove(item)

        if bird.y + bird.image.get_height() >= 730:
            pass
        
        base.move()
        draw_window(window, bird, pipes, base, score)
    pygame.quit()
    quit()

main()

