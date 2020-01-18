import pygame
import neat
import time
import os

from base import Base
from bird import Bird
from pipe import Pipe

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800

GENERATION = 0

pygame.font.init()

BG_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bg.png")))
STAT_FONT = pygame.font.SysFont("comicsans", 50)



def draw_window(window, birds, pipes, base, score, GENERATION):
    # draw background
    window.blit(BG_IMAGE, (0, 0))

    # draw pipe
    for pipe in pipes:
        pipe.draw(window)

    # draw score text
    text1 = STAT_FONT.render("Score:" + str(score), 1,(255, 255, 255))
    window.blit(text1, (WINDOW_WIDTH - 10 - text1.get_width(), 10))

    # draw gen text
    text2 = STAT_FONT.render("Gen:" + str(GENERATION), 1,(255, 255, 255))
    window.blit(text2, (0, 10))
    
    # draw base
    base.draw(window)

    # draw bird
    for bird in birds:
        bird.draw(window)

    pygame.display.update()

def evaluate_genomes(genomes, config):
    global GENERATION
    GENERATION += 1

    score = 0

    networks = []
    current_genomes = []
    birds = []
    initialize_population_with_neural_networks(genomes, config, networks, birds, current_genomes)
    
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pipes = [Pipe(700)]
    clock = pygame.time.Clock()
    
    run = True
    while run and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_index = 0
        if len(birds) > 0:
            if more_than_1_pipes_and_bird_passed_the_first_pipe(birds, pipes):
                pipe_index = 1  # set to next pipe index

        for bird in birds:
            bird.move()
            output = get_output_from_neural_network(bird, pipes, pipe_index, networks, birds)
            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        pipe_to_remove = []
        for pipe in pipes:
            for bird in birds:
                if pipe.collide(bird):
                    remove_birds(current_genomes, birds, bird, networks)
            
                if bird_passed_pipe(bird, pipe):
                    pipe.passed = True
                    add_pipe = True
                    score += 1

                    evaluate_genome(current_genomes)

            if pipe_passed_window(pipe):
                pipe_to_remove.append(pipe)
                
            pipe.move()
        
        add_new_pipe_if_bird_passed(add_pipe, pipes)
        remove_passed_pipes(pipe_to_remove, pipes)
        remove_overfly_or_fell_birds(birds, networks, current_genomes)
        
        base = Base(730)
        base.move()
        draw_window(window, birds, pipes, base, score, GENERATION)


def get_output_from_neural_network(bird, pipes, pipe_index, networks, birds):
    top_pipe_offset = abs(bird.y - pipes[pipe_index].height)
    bottom_pipe_offset = abs(bird.y - pipes[pipe_index].bottom)
    output = networks[birds.index(bird)].activate((bird.y, top_pipe_offset, bottom_pipe_offset))
    return output

def evaluate_genome(current_genomes):
    for genome in current_genomes:
        genome.fitness += 1

def more_than_1_pipes_and_bird_passed_the_first_pipe(birds, pipes):
    return len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width()

def add_new_pipe_if_bird_passed(add_pipe, pipes):
    if add_pipe:
        pipes.append(Pipe(600))

def pipe_passed_window(pipe):
    return pipe.x + pipe.PIPE_TOP.get_width() < 0

def initialize_population_with_neural_networks(genomes, config, networks, birds, current_genomes):
    for genome_id, genome in genomes:
        genome.fitness = 0
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        networks.append(network)
        birds.append(Bird(250, 350))
        current_genomes.append(genome)

def bird_passed_pipe(bird, pipe):
    return not pipe.passed and pipe.x < bird.x

def remove_passed_pipes(pipe_to_remove, pipes):
    for pipe in pipe_to_remove:
        pipes.remove(pipe)

def remove_overfly_or_fell_birds(birds, networks, current_genomes):
    for bird in birds:
        if bird.y + bird.image.get_height() >= 730 or bird.y < -50: # if bird flew to high or fell
            remove_birds(networks, birds, bird, current_genomes)

def remove_birds(networks, birds, bird, current_genomes):
    networks.pop(birds.index(bird))
    current_genomes.pop(birds.index(bird))
    birds.pop(birds.index(bird))


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(evaluate_genomes, 100)
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)

