import pygame, time, ctypes, sys
from EventHandler import EventHandler
from Lander import Lander
from Controller import Controller
from Vector import Vector
from GameLogic import GameLogic
from Surface import Surface
from MainMenu import MainMenu
from ResultMenu import ResultMenu
from DataCollection import DataCollection
from NeuralNetHolder import NeuralNetHolder

# ====== NN -> Controller tuning (hybrid autopilot, refined) ======
THRUST_ON        = 0.25   # NN thrust threshold (NN zayıfsa da gaz verebilsin)
TURN_LEFT_T      = 0.40   # NN turn deadzone lower bound
TURN_RIGHT_T     = 0.60   # NN turn deadzone upper bound

FALL_BOOST_VY    = 0.24   # acil thrust için düşüş hızı eşiği
Y_PD_KP          = 0.0012 # dikey P kazancı (yumuşak)
Y_PD_OFFSET      = 0.045  # taban yukarı-ofset (inişe izin ver)

MAX_TILT_DEG     = 30     # yumuşak açı sınırı

# ---- Flare / yaklaşım mantığı ----
Y_FLARE_PX       = 140.0  # pad'e yakınız: flare bölgesi
VY_DESCENT_SLOW  = 0.10   # flare içinde hedef iniş hızı (~px/frame)
VY_CUT_WINDOW    = 0.06   # |v_y| < bu ise thrust'ı tamamen kes
Y_SNAP_PX        = 18.0   # çok yakında: gazı kes, bırak (touchdown)

# Histerezis: thrust açma/kapama farklı eşikler
THRUST_ON_HYST   = 0.02   # vy_target aşımı üzerine ekstra marj ile aç
THRUST_OFF_HYST  = 0.06   # vy_target altına bu kadar inince kapat
PRINT_EVERY      = 15
# ================================================================

class GameLoop:

    def __init__(self):
        self.controller = Controller()
        self.Handler = EventHandler(self.controller)
        self.object_list = []
        self.game_logic = GameLogic()
        self.fps_clock = pygame.time.Clock()
        self.fps = 60
        self.neuralnet = NeuralNetHolder()
        self.version = "v1.03"
        self.prediction_cycle = 0
        self.ap_thrust_on = False   # thrust histerezis durumu

    def init(self, config_data):
        pygame.init()
        if config_data["FULLSCREEN"] == "TRUE":
            user32 = ctypes.windll.user32
            config_data['SCREEN_HEIGHT'] = int(user32.GetSystemMetrics(1))
            config_data['SCREEN_WIDTH']  = int(user32.GetSystemMetrics(0))
            self.screen = pygame.display.set_mode(
                (config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']),
                pygame.FULLSCREEN
            )
        else:
            config_data['SCREEN_HEIGHT'] = int(config_data['SCREEN_HEIGHT'])
            config_data['SCREEN_WIDTH']  = int(config_data['SCREEN_WIDTH'])
            self.screen = pygame.display.set_mode(
                (config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT'])
            )
        pygame.display.set_caption('CE889 Assignment Template')
        pygame.display.set_icon(pygame.image.load(config_data['LANDER_IMG_PATH']))

    def score_calculation(self):
        score = 1000.0 - (self.surface.centre_landing_pad[0] - self.lander.position.x)
        angle = self.lander.current_angle
        if self.lander.current_angle == 0:
            angle = 1
        if self.lander.current_angle > 180:
            angle = abs(self.lander.current_angle - 360)
        score = score / angle
        velocity = 500 - (self.lander.velocity.x + self.lander.velocity.y)
        score = score + velocity
        print("lander difference " + str(self.surface.centre_landing_pad[0] - self.lander.position.x))
        print("SCORE " + str(score))
        return score

    def main_loop(self, config_data):
        pygame.font.init()
        myfont = pygame.font.SysFont('Comic Sans MS', 30)

        sprites = pygame.sprite.Group()

        on_menus = [True, False, False]  # Main, Won, Lost
        game_start = False

        # Game modes: Play Game, Data Collection, Neural Net, Quit
        game_modes = [False, False, False, False]

        background_image = pygame.image.load(config_data['BACKGROUND_IMG_PATH']).convert_alpha()
        background_image = pygame.transform.scale(
            background_image, (config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT'])
        )

        data_collector = DataCollection(config_data["ALL_DATA"])
        main_menu = MainMenu((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))
        result_menu = ResultMenu((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))
        score = 0

        while True:
            if game_modes[-1]:
                pygame.quit()
                sys.exit()

            if game_start:
                self.controller = Controller()
                self.Handler = EventHandler(self.controller)
                sprites = pygame.sprite.Group()
                self.game_start(config_data, sprites)
                self.ap_thrust_on = False  # reset hysteresis each attempt

            if on_menus[0] or on_menus[1] or on_menus[2]:
                if on_menus[1] or on_menus[2]:
                    result_menu.draw_result_objects(self.screen, on_menus[1], score)
                else:
                    main_menu.draw_buttons(self.screen)
                    textsurface = myfont.render(self.version, False, (0, 0, 0))
                    self.screen.blit(textsurface, (0, 0))

                for event in pygame.event.get():
                    if on_menus[0]:
                        main_menu.check_hover(event)
                        button_clicked = main_menu.check_button_click(event)
                        main_menu.draw_buttons(self.screen)
                        if button_clicked > -1:
                            game_modes[button_clicked] = True
                            on_menus[0] = False
                            game_start = True
                    else:
                        result_menu.check_hover(event)
                        on_menus[0] = result_menu.check_back_main_menu(event)
                        result_menu.draw_result_objects(self.screen, on_menus[1], score)
                        if on_menus[0]:
                            on_menus[1] = False
                            on_menus[2] = False
            else:
                self.Handler.handle(pygame.event.get())

                # ------------------ Neural Net mode (HYBRID + PAD-AWARE) ------------------
                if game_modes[2]:
                    self.prediction_cycle = (self.prediction_cycle + 1) % 2
                    if self.prediction_cycle >= 0:  # =0 yaparsan her 2 frameden bir çalışır
                        input_row = data_collector.get_input_row(self.lander, self.surface, self.controller)

                        # read distances (list/dict tolerant)
                        if isinstance(input_row, dict):
                            x_dist_raw = input_row.get("x_dist", 0.0)
                            y_dist_raw = input_row.get("y_dist", 0.0)
                        else:
                            x_dist_raw = input_row[0] if len(input_row) > 0 else 0.0
                            y_dist_raw = input_row[1] if len(input_row) > 1 else 0.0

                        try:  x_dist = float(str(x_dist_raw).replace(",", "."))
                        except: x_dist = 0.0
                        try:  y_dist = float(str(y_dist_raw).replace(",", "."))
                        except: y_dist = 0.0

                        # NN prediction in [0,1]
                        thrust_hat, turn_hat = self.neuralnet.predict([x_dist, y_dist])

                        # reset controls
                        self.controller.set_up(False)
                        self.controller.set_left(False)
                        self.controller.set_right(False)

                        # ---------- Vertical control: PD + flare + hysteresis ----------
                        vy = float(self.lander.velocity.y)  # + is down
                        vy_target = -min(0.30, Y_PD_KP * abs(y_dist) + Y_PD_OFFSET)
                        if abs(y_dist) < Y_FLARE_PX:   # flare region
                            vy_target = -VY_DESCENT_SLOW

                        # touchdown window: cut thrust if we're slow & close
                        if abs(y_dist) < Y_SNAP_PX and abs(vy) < VY_CUT_WINDOW:
                            self.ap_thrust_on = False
                        else:
                            if vy > FALL_BOOST_VY:
                                self.ap_thrust_on = True
                            elif (vy > vy_target + THRUST_ON_HYST) or (thrust_hat > THRUST_ON):
                                self.ap_thrust_on = True
                            elif (vy < vy_target - THRUST_OFF_HYST):
                                self.ap_thrust_on = False
                            # else keep state

                        if self.ap_thrust_on:
                            self.controller.set_up(True)

                        # ---------- Horizontal centering: PAD-AWARE PD ----------
                        # Pad gerçekten haritanın ortasında değilse bile buradan öğreniyoruz:
                        pad_x = float(self.surface.centre_landing_pad[0])
                        x_now = float(self.lander.position.x)
                        vx    = float(self.lander.velocity.x)

                        # predicted error = position error - Kv * velocity  (overshoot'u azaltır)
                        K_v   = 12.0                      # 8–18 arası deneyebilirsin
                        e_x   = pad_x - x_now              # sağ +, sol −
                        e_pred = e_x - K_v * vx

                        # dinamik koridor: yere indikçe daralsın
                        DEAD_MIN, DEAD_MAX, DEAD_SLOPE = 6.0, 40.0, 0.06
                        dead = max(DEAD_MIN, min(DEAD_MAX, DEAD_SLOPE * abs(y_dist)))

                        if   e_pred >  dead:
                            self.controller.set_right(True)
                        elif e_pred < -dead:
                            self.controller.set_left(True)
                        else:
                            # koridorda: NN'in turn çıkışını kullan (geniş deadzone)
                            if turn_hat < TURN_LEFT_T:
                                self.controller.set_left(True)
                            elif turn_hat > TURN_RIGHT_T:
                                self.controller.set_right(True)
                            # else: neutral

                        # ---------- Soft tilt clamp ----------
                        ang = self.lander.current_angle
                        if (ang > MAX_TILT_DEG and ang < (360 - MAX_TILT_DEG)):
                            alpha = (ang - MAX_TILT_DEG) / (360 - 2 * MAX_TILT_DEG)
                            alpha = round(alpha)
                            if alpha == 0:
                                self.lander.current_angle = MAX_TILT_DEG
                            else:
                                self.lander.current_angle = 360 - MAX_TILT_DEG

                        # debug
                        if (pygame.time.get_ticks() // (1000 // self.fps)) % PRINT_EVERY == 0:
                            print(f"[NN] x={x_dist:.1f} y={y_dist:.1f} | out(t,τ)=({thrust_hat:.2f},{turn_hat:.2f}) "
                                  f"| vx={vx:.2f} vy={vy:.2f} | pad_x={pad_x:.1f} e_x={e_x:.1f} dead={dead:.1f} "
                                  f"| up={self.controller.up} L={self.controller.left} R={self.controller.right}")
                # -----------------------------------------------------------------------

                self.screen.blit(background_image, (0, 0))
                if (not self.Handler.first_key_press) and game_start:
                    self.update_objects()
                    game_start = False

                if self.Handler.first_key_press:
                    data_input_row = data_collector.get_input_row(self.lander, self.surface, self.controller)
                    self.update_objects()
                    if game_modes[1]:
                        data_collector.save_current_status(data_input_row, self.lander, self.surface, self.controller)

                sprites.draw(self.screen)

                # win / lose checks
                if self.lander.landing_pad_collision(self.surface):
                    score = self.score_calculation()
                    on_menus[1] = True
                    if game_modes[1]:
                        data_collector.write_to_file()
                        data_collector.reset()
                elif (self.lander.surface_collision(self.surface) or
                      self.lander.window_collision((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))):
                    on_menus[2] = True
                    data_collector.reset()

                if on_menus[1] or on_menus[2]:
                    game_start = False
                    for i in range(len(game_modes)):
                        game_modes[i] = False

            pygame.display.flip()
            self.fps_clock.tick(self.fps)

    def update_objects(self):
        self.game_logic.update(0.2)

    def setup_lander(self, config_data):
        lander = Lander(
            config_data['LANDER_IMG_PATH'],
            [config_data['SCREEN_WIDTH'] / 2, config_data['SCREEN_HEIGHT'] / 2],
            Vector(0, 0),
            self.controller
        )
        self.game_logic.add_lander(lander)
        return lander

    def game_start(self, config_data, sprites):
        self.lander = self.setup_lander(config_data)
        self.surface = Surface((config_data['SCREEN_WIDTH'], config_data['SCREEN_HEIGHT']))
        sprites.add(self.lander)
        sprites.add(self.surface)
