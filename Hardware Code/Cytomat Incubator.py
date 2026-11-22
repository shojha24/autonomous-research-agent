# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:04:01 2024

@author: Core 4
"""
import serial
import time

class IncubatorCytomat:

    # Constructor, connects to Cytomat Incubator via serial port
    def __init__(self, port="COM1"):
        self.connection = serial.Serial(port=port, 
                            baudrate=9600, 
                            bytesize=serial.EIGHTBITS, 
                            parity=serial.PARITY_NONE, 
                            stopbits=serial.STOPBITS_ONE, timeout=1)
        # Cytomat command to initialize the physical incubator 
        self.send(f"ll:in")
        
    # Destructor 
    def __del__(self):
        if self.connection.is_open:
            self.connection.close()
        else:
            print('tried to close a connection that was already closed')

    # Import cell plate from transfer station into specific rack and slot of incubator  
    def import_plate(self, incu_rack, incu_slot):
        # imports plate from transfer station in front of handler into the desired incubator location
        
        # check valid rack and slot inputs
        if incu_rack < 1 or incu_rack > 2 or incu_slot < 1 or incu_slot > 21:
            raise ValueError("Invalid incubator rack and slot.")
        
        # convert inputs to cytomat numbering format
        if incu_rack == 1 and incu_slot < 10:  
            three_digits = f"00{incu_slot}"
        elif incu_rack == 1 and incu_slot >= 10: 
            three_digits = f"0{incu_slot}"
        elif incu_rack == 2:
            three_digits = f"0{21 + incu_slot}"
        else:
            raise ValueError("Invalid incubator rack and slot.")
        
        # check is_busy for 15 seconds 
        max_wait_time = 15
        wait_time = 0
        while self.is_busy() and wait_time < max_wait_time:
            if wait_time == 0: 
                print("Waiting for busy status to be false")
            time.sleep(0.1)
            wait_time += 0.1
        if wait_time > max_wait_time:
            raise ValueError("Program stopped. Status still busy.")

        # send command
        resp = self.send(f"mv:ts {three_digits}")
        
        # get response
        max_program_wait_time = 30
        print(self.command_response(resp, max_program_wait_time, 'import plate'))
        
        
    # Export plate from a specific rack and slot onto the transfer station
    def export_plate(self, incu_rack, incu_slot):
        # check valid rack and slot inputs
       if incu_rack < 1 or incu_rack > 2 or incu_slot < 1 or incu_slot > 21:
           raise ValueError("Invalid incubator rack and slot.")
    
       # convert inputs to cytomat numbering format
       if incu_rack == 1 and incu_slot < 10:  
           three_digits = f"00{incu_slot}"
       elif incu_rack == 1 and incu_slot >= 10: 
           three_digits = f"0{incu_slot}"
       elif incu_rack == 2:
           three_digits = f"0{21 + incu_slot}"
       else:
           raise ValueError("Invalid incubator rack and slot.")
       
       # check is_busy for 15 seconds
       max_wait_time = 15
       wait_time = 0
       while self.is_busy() and wait_time < max_wait_time:
            if wait_time == 0: 
                print("Waiting for busy status to be false")
            time.sleep(0.1)
            wait_time += 0.1
       if wait_time > max_wait_time:
            raise ValueError("Program stopped. Status still busy.") 
        
       # send command
       resp = self.send(f"mv:st {three_digits}")
           
       # get response
       max_program_wait_time = 30
       print(self.command_response(resp, max_program_wait_time, 'export plate'))
       
       
    # Swaps transfer station 
    def swap_transfer(self):
        # check is_busy for 15 seconds 
        max_wait_time = 15
        wait_time = 0
        check_interval_sec = 0.25
        while self.is_busy() and wait_time < max_wait_time:
            if wait_time == 0: 
                print("Waiting for busy status to be false")
            time.sleep(check_interval_sec)
            wait_time += check_interval_sec

        if wait_time > max_wait_time:
            raise ValueError("Program stopped. Status still busy.")
            
        # check current position of swap station
        swap_position = int(self.send("ch:sw")[3])
        max_program_wait_time = 15
        
        # send command
        if swap_position == 1:
            resp = self.send("ll:tp 002")
            print(self.command_response(resp, max_program_wait_time, 'swap transfer'))
        elif swap_position == 2:
            resp = self.send("ll:tp 001")
            print(self.command_response(resp, max_program_wait_time, 'swap transfer'))
        else: return False
        return True
    
    # Initialize physical incubator  
    def initialize(self):
         self.send("ll:in")
         pass
    
    def recover_plate_in_handler(self):
        # returns plate to transfer station 1
        
        # check is_busy for 15 seconds 
        max_wait_time = 15
        wait_time = 0
        check_interval_sec = 0.25
        while self.is_busy() and wait_time < max_wait_time:
            if wait_time == 0: 
                print("Waiting for busy status to be false")
            time.sleep(check_interval_sec)
            wait_time += check_interval_sec
        
        # ensure that transfer station 1 is in front of robotic arm
        if int(self.send("ch:sw")[3]) == 1:
            self.swap_transfer()
        
        # send command to move plate from wait position to the transfer station 
        resp = self.send("mv:wt")
        max_program_wait_time = 15
        print(self.command_response(resp, max_program_wait_time, 'recover plate in handler'))

    # Sends commands to incubator 
    def send(self, msg):
        self.connection.write(f'{msg}\r'.encode('utf-8'))
        n_iter, max_iter = 0, 100
        time.sleep(0.1)
        while not self.connection.in_waiting and n_iter < max_iter:
            time.sleep(0.1)
            n_iter += 1
        
        if self.connection.in_waiting:
            resp = self.connection.read(self.connection.in_waiting)
        else:
            raise NotImplementedError("No response in time")
            
        return resp.decode('utf-8')
    
    # Prints status of incubator 
    def status(self, to_print = True):

        # Send command to receive status of incubator 
        out = self.send("ch:bs")
        if to_print: print('out: ', out)
        try:
            l_digit = int(out[3])
        except:
            letters = ['A', 'B', 'C', 'D', 'E', 'F']
            for letter in letters:
                l_values = [10, 11, 12, 13, 14, 15]
                if out[3] == letter:
                    l_digit = l_values[letters.index(letter)]     

        # Algorithm to convert binary 8-bit binary to the associated label 
        r_digit = int(out[4])
        l_labels = ['Handler occupied', 'Gate open', 'Device door open', 'Transfer station occupied']
        r_labels = ['Busy set', 'Ready set', 'Warning register set', 'Error register set']
        output_bools = []
        for label, bit in zip(l_labels, [1,2,4,8]):
            flag = l_digit & bit != 0
            output_bools.append(flag)
            if to_print: print(flag, '\t', label)
        for label, bit in zip(r_labels, [1,2,4,8]):
            flag = r_digit & bit != 0
            output_bools.append(flag)
            if to_print: print(flag, '\t', label)
        return output_bools
    
    # Returns if incubator is ready 
    def is_ready(self):
        return self.status()[5]
    
    # Returns if incubaor is busy
    def is_busy(self):
        if self.status(to_print = False)[4]:
            return True
        else: return False
        
    # Prints if the incubator has a warning
    def is_warning(self):
        if self.status(to_print = False)[6]:
            warning = self.send('ch:bw')
            warning_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 'C']
            warning_messages = ["Communication with motor controllers interrupted",
                            "No microplate loaded on handler/shovel",
                            "No microplate unloaded from handler/shovel",
                            "Shovel not extended/Automatic unit position error",
                            "Process timeout",
                            "Automatic lift door not open",
                            "Automatic lift door not closed",
                            "Shovel not retracted",
                            "Initialization due to open device door",
                            "Transfer station did not rotate"]
            for warn in warning_values:
                if warn == warning[3]: 
                    return (True , f'Unable to execute command: {warning_messages[warning_values.index(warn)]}')
        return (False, None)
    
    # Prints if the incubator has an error
    def is_error(self):
        # return (True, self.send('ch:be'))
        if self.status(to_print = False)[7]:
            error = self.send('ch:be')
            error_values = [1, 2, 3, 4, 5, 6, 7, 8, 'A', 'B', 'C', 'F']
            error_messages = ["Communication with motor controllers interrupted",
                            "No microplate loaded on handler/shovel",
                            "No microplate unloaded from handler/shovel",
                            "Shovel not extended/Automatic unit position error",
                            "Process timeout",
                            "Automatic lift door not open",
                            "Automatic lift door not closed",
                            "Shovel not retracted",
                            "Stepper motor controller temperature too high",
                            "Communication with heating controllers and CO2 supply",
                            "Transfer station not rotated",
                            "Fatal error occurred during error routine"]
            for er in error_values:
                if er == int(error[4]): 
                    return (True , f'Register error: {error_messages[error_values.index(er)]}')
        else: return (False, None)
    
    # Command to reset error status 
    def reset_error(self):
       return self.send("rs:be")
    
    # Returns the response that an incubator sends after receiving a command 
    def command_response(self, resp, max_wait_time, command):
        # if 'err' is sent in
        if resp[:2] != 'ok':
            error_val = resp[3:]
            error_values = [1, 2, 3, 4, 5, 11, 12, 21, 22, 31, 32, 33, 41, 42, 51, 52]
            error_messages = ["Device still busy, new command not accepted", "Unknown error", 
                              "Telegram structure error", "Incorrect parameters in telegram", 
                              "Uknown location number specified", "Incorrect handler position", 
                              "Command not executable as shovel is extended", "Handler already occupied",
                              "Handler empty", "Transfer station empty", "Transfer station occupied", 
                              "Transfer station not in position", "No automatic lift door configured",
                              "Automatic lift door not open", "Error while accessing internal memory",
                              "Incorrect password/unauthorized access"]
            for er in error_values:
                if er == error_val: 
                    return (f'Unable to execute {command}: {error_messages[error_values.index(er)]}')

        # if 'ok' is sent in
        wait_time = 0
        check_interval_sec = 0.5
        while self.is_busy() and wait_time < max_wait_time:
            time.sleep(check_interval_sec)
            wait_time += check_interval_sec
            # check warning status
            self.is_warning()

        if wait_time > max_wait_time:
            raise ValueError("Program stopped. Exceded waiting time.")
        
        # if program is completed, check error status
        if not (self.is_busy() and self.is_error()):
            return (True, f'{command} executed without error.')
        else:
            return(False, f"{command} executed with error: {self.is_error()[1]}")


    #
    # integrated_import_plate sends a microplate from the transfer station in front of the robotic arm to
    #   the desired incubator slot
    #   
    def integrated_import_plate(self, incu_rack, incu_slot):
        
        # if there is already a plate on transfer station 1 in front of the gate, simply import the plate
        if int(self.send("ch:sw")[3]) == 1 and int(self.send("ch:sw")[4]) == 1:
            self.import_plate(incu_rack, incu_slot)
            return
            
        # if plate mistakenly placed on transfer station 2, the program stops with an error
        if int(self.send("ch:sw")[3]) == 2:
            if int(self.send("ch:sw")[4]) == 1:
                raise ValueError("Program stopped. Plate should be placed on Transfer Station 1.")
        else:
            if int(self.send("ch:sw")[5]) == 1:
                raise ValueError("Program stopped. Plate should be placed on Transfer Station 1.")
        
        # ensure that transfer station 1 is in front of robotic arm
        if int(self.send("ch:sw")[3]) == 1:
            self.swap_transfer()
            
        # wait for plate to be loaded on to transfer station 1
        max_wait_time = 15
        wait_time = 0
        while wait_time < max_wait_time:
            if int(self.send("ch:sw")[5]) == 1: 
                break
            time.sleep(0.5)
            wait_time += 0.5

        if wait_time > max_wait_time - 1:
            raise ValueError("Program stopped. No plate detected.")
        time.sleep(1)
        
        # swap transfer to be in front of the gate
        self.swap_transfer()
        # if plate is removed anytime from the swap process, 
        #   the program waits 10 seconds for a plate to be placed back on swap transfer 1
        #   if no plate is put after 10 seconds, the program stops with an error
        time.sleep(1)
        if int(self.send("ch:sw")[4]) == 0:
            print("no plate on transfer")
            while wait_time < 10:
                if int(self.send("ch:sw")[4]) == 1: 
                    time.sleep(1)
                    break
                time.sleep(0.5)
                wait_time += 0.5
                
            if wait_time > 9:
                raise ValueError("Program stopped. No plate detected. Run command again when ready.")
        
        # execute import plate commmand
        self.import_plate(incu_rack, incu_slot)
        

    #
    # integrated_export_plate retrieves a microplate and sends it to the transfer station 
    #   in front of the robotic arm 
    #
    def integrated_export_plate(self, incu_rack, incu_slot):
        
        time.sleep(2)
        
        # ensure that transfer station 1 is in front of the gate
        if int(self.send("ch:sw")[3]) == 2:
            self.swap_transfer()
        
        # swap transfer and execute import plate function 
        self.export_plate(incu_rack, incu_slot)
        
        self.swap_transfer()
        # if plate is removed anytime from the swap process, 
        #   the program waits 10 seconds for a plate to be placed back on swap transfer 1
        #   if no plate is put after 10 seconds, the program stops with an error
        wait_time = 0
        if int(self.send("ch:sw")[5]) == 0:
            while wait_time < 5:
                if int(self.send("ch:sw")[5]) == 1: 
                    break
                time.sleep(0.5)
                wait_time += 0.5
                
            if wait_time > 5:
                raise ValueError("Note: No plate detected on transfer station")
        
    
    def read_temp(self):     
        pass
    
    def set_temp(self):
        pass
    
    def read_co2(self):
        pass
    
    def set_co2(self):
        pass
            
        
if __name__ == "__main__":
    ic = IncubatorCytomat()
