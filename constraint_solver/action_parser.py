class ActionParser():
    def __init__(self):
        self.actions = []

    def parse(self, text):
        lines = text.split("\n")
        description = self.locate_actions(lines)

        for line in description:
            self.parse_action(line)

        return self.actions
    
    def locate_actions(self, lines):
        for i, line in enumerate(lines):
            line = line.strip()
            if line == "<Start Subsequent Actions>":
                start = i
            elif line == "<Start Subsequent Actions>":
                end = i
        return lines[start+1:end]
    
    def parse_action(self, line):
        line = line.strip().strip('.').lower()
        words = line.replace('\'', ' ').split()
        if "move" in line:
            if "left" in line:
                action = "move_left"
            elif "right" in line:
                action = "move_right"
            elif "forward" in line:
                action = "move_forward"
            elif "backward" in line:
                action = "move_backward"
            elif "up" in line:
                action = "move_up"
            elif "down" in line:
                action = "move_down"
            for i, word in enumerate(words):
                if word == 'cm':
                    distance = float(words[i-1]) / 100
            self.actions.append((action, distance))
        elif "rotate" in line:
            action = "rotate"
            for i, word in enumerate(words):
                if word == 'degree':
                    angle = float(words[i-1])
            self.actions.append((action, angle))
        elif "open" in line:
            self.actions.append(("open", 0))
        elif "close" in line:
            self.actions.append(("close", 0))
            