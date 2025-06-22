import pygame 
pygame.init()



def create_text_render(string):
    font = pygame.font.SysFont('sans', 40)
    return font.render(string, True, WHITE)


screen = pygame.display.set_mode(((1200, 700)))

pygame.display.set_caption("Kmeans Classification")

running = True

clock = pygame.time.Clock()
BACKGROUND = (214, 214, 214)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BACKGROUND_PANEL = (249, 255, 230)

font = pygame.font.SysFont('sans', 40)
# text_plus = font.render('+', True, WHITE)
# text_minus = font.render('+', True, WHITE)

K = 0
error = 0
while running:
    clock.tick(60) # 60 FPS
    screen.fill(BACKGROUND)
    # Draw interface

    # Draw panel

    pygame.draw.rect(screen, BLACK, (50, 50, 700, 500)) # x1,y1 = (50, 50), x2,y2 = (700, 500)
    pygame.draw.rect(screen, BACKGROUND_PANEL, (55, 55,690, 490))

    # K button + 
    pygame.draw.rect(screen, BLACK, (850, 50, 50, 50))
    screen.blit(create_text_render("+"), (860, 50))

    # K button -

    pygame.draw.rect(screen, BLACK, (950, 50, 50, 50))
    screen.blit(create_text_render("-"), (960, 50))

    # K value 
    text_K = font.render("K = " + str(K), True, BLACK)
    screen.blit(text_K, (1050, 50))

    # Run button
    pygame.draw.rect(screen, BLACK, (850, 150, 150, 50))
    screen.blit(create_text_render("Run"), (900, 150))

    # Random button
    pygame.draw.rect(screen, BLACK, (850, 250, 150, 50))
    screen.blit(create_text_render("Random"), (850, 250))

    # Reset button 
    pygame.draw.rect(screen, BLACK, (850, 550, 150, 50))
    screen.blit(create_text_render("Reset"), (850, 550))

    # Algorithm button 
    pygame.draw.rect(screen, BLACK, (850, 450, 150, 50))
    screen.blit(create_text_render("Algorithm"), (850, 450))

    # Error text
    text_error = font.render("Error = " + str(int(error)), True, BLACK)
    screen.blit(text_error, (850, 350))




    # End draw interface

    mouse_x , mouse_y = pygame.mouse.get_pos()
    
    

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Change K button + 
            if 850 < mouse_x < 900  and 50 < mouse_y < 100:
                K = K + 1
                print("press K+")
            
            # Change K button -
            if 950 < mouse_x < 1000  and 50 < mouse_y < 100:
                if K > 0:
                    K -= 1
                print("press K-")
            # Run button 

            if 850 < mouse_x < 1000 and 150 < mouse_y < 200:
                print("run pressed")


            # Random button

            if 850 < mouse_x < 1000 and 250 < mouse_y < 300:
                print("random pressed")
            
            # Reset button
            if 850 < mouse_x < 1000 and 550 < mouse_y < 600:
                print("Reset button pressed")
            
            # Algorithm
            if 850 < mouse_x < 1000 and 450 < mouse_y < 500:
                print("Algorithm button pressed")



    
    pygame.display.flip()



pygame.quit()