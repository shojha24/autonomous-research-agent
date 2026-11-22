SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

CREATE SCHEMA IF NOT EXISTS `Experiment_Automation` DEFAULT CHARACTER SET utf8;
USE `Experiment_Automation`;

-- -----------------------------------------------------
-- machines_inventory 
-- 		lists all the machines that can be used to perform actions.
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`machines_inventory` (
  `machine_id` INT NOT NULL,
  `machine_description` TEXT NULL,
  PRIMARY KEY (`machine_id`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- action_inventory
-- 		"action_inventory" is a bank of all the possible actions 
-- 		that can be done to a sample, cell well, plate, incubator, 
-- 		media, and reagent. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`action_inventory` (
  `action_id` INT NOT NULL,
  `description` TEXT NULL,
  `machine_id` INT NULL,
  PRIMARY KEY (`action_id`),
  INDEX `fk_action_inventory_machines_inventory1_idx` (`machine_id` ASC) VISIBLE,
  CONSTRAINT `fk_action_inventory_machines_inventory1`
    FOREIGN KEY (`machine_id`)
    REFERENCES `Experiment_Automation`.`machines_inventory` (`machine_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- photo_inventory
-- 		"photo_inventory" houses all photos of cell samples. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`photo_inventory` (
  `photo_id` INT NOT NULL,
  `link` TEXT NULL,
  `description` TEXT NULL,
  PRIMARY KEY (`photo_id`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- media_properties
-- 		"media_properties" creates an object for every type 
-- 		of media used in experiements for documentation and 
-- 		to ensure proper storage and working environemnt. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`media_properties` (
  `media_name` VARCHAR(100) NOT NULL,
  `lot_number` INT NULL,
  `storage_temperature` DECIMAL(6,2) NULL,
  `working_temperature` DECIMAL(6,2) NULL,
  PRIMARY KEY (`media_name`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- reagent_properties
-- 		"reagent_properties" creates an object for every type
-- 		of media used in experiements documentation and to ensure 
-- 		proper storage and working environemnt. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`reagent_properties` (
  `reagent_name` VARCHAR(100) NOT NULL,
  `lot_number` INT NULL,
  `storage_temperature` DECIMAL(6,2) NULL,
  `working_temperature` DECIMAL(6,2) NULL,
  PRIMARY KEY (`reagent_name`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- cell_properties
-- 		"cell_properties" contains relevant attributes of 
-- 		a specific cell and cell clone that can be used to 
-- 		inform how much media/reagents is to be used during culture, 
-- 		the frequency of feeding, and its ideal storage conditions. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`cell_properties` (
  `cell_name` VARCHAR(100) NOT NULL,
  `cell_clone` VARCHAR(100) NOT NULL,
  `cell_type` VARCHAR(100) NULL,
  `reagent_name` VARCHAR(100) NULL,
  `media_name` VARCHAR(100) NULL,
  `ideal_co2` DECIMAL(6,2) NULL,
  `ideal_temp` DECIMAL(6,2) NULL,
  `photo_id` INT NULL,
  `max_passage` INT NULL,
  `doubling_time` DECIMAL(6,2) NULL,
  `confluency_frequency` DECIMAL(6,2) NULL,
  `media_frequency` DECIMAL(6,2) NULL,
  PRIMARY KEY (`cell_name`, `cell_clone`),
  INDEX `fk_cell_properties_photo_inventory1_idx` (`photo_id` ASC) VISIBLE,
  INDEX `fk_cell_properties_media_properties1_idx` (`media_name` ASC) VISIBLE,
  INDEX `fk_cell_properties_reagent_properties1_idx` (`reagent_name` ASC) VISIBLE,
  CONSTRAINT `fk_cell_properties_photo_inventory1`
    FOREIGN KEY (`photo_id`)
    REFERENCES `Experiment_Automation`.`photo_inventory` (`photo_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_cell_properties_media_properties1`
    FOREIGN KEY (`media_name`)
    REFERENCES `Experiment_Automation`.`media_properties` (`media_name`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_cell_properties_reagent_properties1`
    FOREIGN KEY (`reagent_name`)
    REFERENCES `Experiment_Automation`.`reagent_properties` (`reagent_name`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- well_confluency_tracker
-- 		"well_confluency_tracker" tracks the confluency of all wells over time.
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`well_confluency_tracker` (
  `well_id` INT NOT NULL,
  `check_date_time` DATETIME NOT NULL,
  `confluency` DECIMAL(6,2) NULL,
  PRIMARY KEY (`well_id`, `check_date_time`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- well_actions_tracker
-- 		"well_actions_tracker" records any changes or actions
-- 		done to any well during an experiment. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`well_actions_tracker` (
  `date_time` DATETIME NOT NULL,
  `well_id` INT NOT NULL,
  `action_id` INT NULL,
  `notes` TEXT NULL,
  PRIMARY KEY (`date_time`, `well_id`),
  INDEX `fk_well_actions_tracker_action_inventory1_idx` (`action_id` ASC) VISIBLE,
  CONSTRAINT `fk_well_actions_tracker_action_inventory1`
    FOREIGN KEY (`action_id`)
    REFERENCES `Experiment_Automation`.`action_inventory` (`action_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- well_inventory
-- 		"""well_inventory"" keeps track of the properties of every 
-- 		cell culture well that is currently or has been previously 
-- 		used in an experiment. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`well_inventory` (
  `well_id` INT NOT NULL,
  `plate_id` INT NULL,
  `sample_id` INT NULL,
  `plate_row` INT NULL,
  `plate_column` INT NULL,
  `previous_well_id` INT NULL,
  `passage_number` INT NULL,
  `current_confluency` DECIMAL(6,2) NULL,
  `media_volume_liter` DECIMAL(6,4) NULL,
  `split_volume_liter` DECIMAL(6,4) NULL,
  `max_well_volume` DECIMAL(6,4) NULL,
  `start_date_time` DATETIME NULL,
  `end_date_time` DATETIME NULL,
  PRIMARY KEY (`well_id`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- sample_inventory
-- 		"""reagent_inventory"" tracks all containers 
-- 		of reagent that is currently or has been previously 
-- 		used in an experiment. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`sample_inventory` (
  `sample_id` INT NOT NULL,
  `cell_name` VARCHAR(100) NULL,
  `cell_clone` VARCHAR(100) NULL,
  `photo_id` INT NULL,
  `current_plate_id` INT NULL,
  `lot_number` INT NULL,
  `start_date_time` DATETIME NULL,
  `end_date_time` DATETIME NULL,
  PRIMARY KEY (`sample_id`),
  INDEX `fk_sample_inventory_cell_properties1_idx` (`cell_name`, `cell_clone`) VISIBLE,
  INDEX `fk_sample_inventory_photo_inventory1_idx` (`photo_id` ASC) VISIBLE,
  CONSTRAINT `fk_sample_inventory_cell_properties1`
    FOREIGN KEY (`cell_name`, `cell_clone`)
    REFERENCES `Experiment_Automation`.`cell_properties` (`cell_name`, `cell_clone`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_sample_inventory_photo_inventory1`
    FOREIGN KEY (`photo_id`)
    REFERENCES `Experiment_Automation`.`photo_inventory` (`photo_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- media_volume_tracker
-- 		"media_volume_tracker" tracks the volume of all media containers over time.  
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`media_volume_tracker` (
  `media_id` INT NOT NULL,
  `check_date_time` DATETIME NOT NULL,
  `volume_liter` DECIMAL(6,4) NULL,
  PRIMARY KEY (`media_id`, `check_date_time`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- media_inventory
-- 		"""media_inventory"" tracks all containers of media 
-- 		that is currently or has been previously used in an experiment. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`media_inventory` (
  `media_id` INT NOT NULL,
  `media_name` VARCHAR(100) NULL,
  `is_in_use` BINARY NULL,
  `incubator_name` VARCHAR(45) NULL,
  `incubator_stack` INT NULL,
  `incubator_slot` INT NULL,
  `current_volume_liter` DECIMAL(6,4) NULL,
  `max_volume_liter` DECIMAL(6,4) NULL,
  `start_date_time` DATETIME NULL,
  `end_date_time` DATETIME NULL,
  PRIMARY KEY (`media_id`),
  INDEX `fk_media_inventory_media_properties1_idx` (`media_name` ASC) VISIBLE,
  CONSTRAINT `fk_media_inventory_media_properties1`
    FOREIGN KEY (`media_name`)
    REFERENCES `Experiment_Automation`.`media_properties` (`media_name`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_media_inventory_media_volume_tracker1`
    FOREIGN KEY (`media_id`)
    REFERENCES `Experiment_Automation`.`media_volume_tracker` (`media_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- reagent_inventory
-- 		"""reagent_inventory"" tracks all containers of 
-- 		reagent that is currently or has been previously used in an experiment. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`reagent_inventory` (
  `reagent_id` INT NOT NULL,
  `reagent_name` VARCHAR(100) NULL,
  `is_in_use` BINARY NULL,
  `incubator_name` VARCHAR(45) NULL,
  `incubator_stack` INT NULL,
  `incubator_slot` INT NULL,
  `current_volume_liter` DECIMAL(6,4) NULL,
  `max_volume_liter` DECIMAL(6,4) NULL,
  `start_date_time` DATETIME NULL,
  `end_date_time` DATETIME NULL,
  PRIMARY KEY (`reagent_id`),
  INDEX `fk_reagent_inventory_reagent_properties1_idx` (`reagent_name` ASC) VISIBLE,
  CONSTRAINT `fk_reagent_inventory_reagent_properties1`
    FOREIGN KEY (`reagent_name`)
    REFERENCES `Experiment_Automation`.`reagent_properties` (`reagent_name`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- reagent_volume_tracker
-- 		"reagent_volume_tracker" tracks the volume of all reagents 
-- 		containers over time.  
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`reagent_volume_tracker` (
  `reagent_id` INT NOT NULL,
  `check_date_time` DATETIME NOT NULL,
  `volume_liter` DECIMAL(6,4) NULL,
  PRIMARY KEY (`reagent_id`, `check_date_time`),
  CONSTRAINT `fk_reagent_volume_tracker_reagent_inventory1`
    FOREIGN KEY (`reagent_id`)
    REFERENCES `Experiment_Automation`.`reagent_inventory` (`reagent_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- incubator_inventory
-- 		"incubator_inventory" keeps track of what is currently in the incubators. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`incubator_inventory` (
  `incubator_name` VARCHAR(45) NOT NULL,
  `incubator_stack` INT NOT NULL,
  `incubator_slot` INT NOT NULL,
  `storage_type` VARCHAR(45) NULL,
  `storage_type_id` INT NULL,
  PRIMARY KEY (`incubator_name`, `incubator_stack`, `incubator_slot`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- plate_inventory
-- 		"plate_inventory" keeps track of all the plates 
-- 		that are currently or previously used in an experiment. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`plate_inventory` (
  `plate_id` INT NOT NULL,
  `plate_barcode` VARCHAR(45) NULL,
  `incubator_name` VARCHAR(45) NULL,
  `incubator_stack` INT NULL,
  `incubator_slot` INT NULL,
  `max_well_count` INT NULL,
  `plate_manufacturer` VARCHAR(100) NULL,
  `catalog_number` INT NULL,
  `start_date_time` DATETIME NULL,
  `end_date_time` DATETIME NULL,
  PRIMARY KEY (`plate_id`),
  INDEX `fk_plate_inventory_incubator_inventory1_idx` (`incubator_name`, `incubator_stack`, `incubator_slot`) VISIBLE,
  CONSTRAINT `fk_plate_inventory_incubator_inventory1`
    FOREIGN KEY (`incubator_name`, `incubator_stack`, `incubator_slot`)
    REFERENCES `Experiment_Automation`.`incubator_inventory` (`incubator_name`, `incubator_stack`, `incubator_slot`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- liquid_handler_actions_tracker
-- 		"liquid_handler_actions_tracker" records any changes 
-- 		or actions done to the liquid handler machine
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`liquid_handler_actions_tracker` (
  `date_time` DATETIME NOT NULL,
  `machine_id` INT NOT NULL,
  `action_id` INT NULL,
  `notes` TEXT NULL,
  PRIMARY KEY (`date_time`, `machine_id`),
  INDEX `fk_liquid_handler_actions_tracker_action_inventory1_idx` (`action_id` ASC) VISIBLE,
  INDEX `fk_liquid_handler_actions_tracker_machines_inventory1_idx` (`machine_id` ASC) VISIBLE,
  CONSTRAINT `fk_liquid_handler_actions_tracker_action_inventory1`
    FOREIGN KEY (`action_id`)
    REFERENCES `Experiment_Automation`.`action_inventory` (`action_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_liquid_handler_actions_tracker_machines_inventory1`
    FOREIGN KEY (`machine_id`)
    REFERENCES `Experiment_Automation`.`machines_inventory` (`machine_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- liquid_handler_inventory
-- 		"liquid_handler_inventory" stores attributes of liquid handler, eg. tips count
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`liquid_handler_inventory` (
  `machine_id` INT NOT NULL,
  `current_tips_count` INT NULL,
  `max_tips_count` INT NULL,
  PRIMARY KEY (`machine_id`),
  CONSTRAINT `fk_liquid_handler_inventory_machines_inventory1`
    FOREIGN KEY (`machine_id`)
    REFERENCES `Experiment_Automation`.`machines_inventory` (`machine_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- reagent_actions_tracker
-- 		"reagent_actions_tracker" records all changes or actions 
-- 		done to any reagent during an experiment. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`reagent_actions_tracker` (
  `date_time` DATETIME NOT NULL,
  `reagent_id` INT NOT NULL,
  `action_id` INT NULL,
  `notes` TEXT NULL,
  PRIMARY KEY (`date_time`, `reagent_id`),
  INDEX `fk_reagent_actions_tracker_reagent_inventory1_idx` (`reagent_id` ASC) VISIBLE,
  INDEX `fk_reagent_actions_tracker_action_inventory1_idx` (`action_id` ASC) VISIBLE,
  CONSTRAINT `fk_reagent_actions_tracker_reagent_inventory1`
    FOREIGN KEY (`reagent_id`)
    REFERENCES `Experiment_Automation`.`reagent_inventory` (`reagent_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_reagent_actions_tracker_action_inventory1`
    FOREIGN KEY (`action_id`)
    REFERENCES `Experiment_Automation`.`action_inventory` (`action_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- media_actions_tracker
-- 		"media_actions_tracker" records all changes or actions 
-- 		done to any media during an experiment. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`media_actions_tracker` (
  `date_time` DATETIME NOT NULL,
  `media_id` INT NOT NULL,
  `action_id` INT NULL,
  `notes` TEXT NULL,
  PRIMARY KEY (`date_time`, `media_id`),
  INDEX `fk_media_actions_tracker_media_inventory1_idx` (`media_id` ASC) VISIBLE,
  INDEX `fk_media_actions_tracker_action_inventory1_idx` (`action_id` ASC) VISIBLE,
  CONSTRAINT `fk_media_actions_tracker_media_inventory1`
    FOREIGN KEY (`media_id`)
    REFERENCES `Experiment_Automation`.`media_inventory` (`media_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_media_actions_tracker_action_inventory1`
    FOREIGN KEY (`action_id`)
    REFERENCES `Experiment_Automation`.`action_inventory` (`action_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- sample_actions_tracker
-- 		"sample_actions_tracker" records all changes or 
-- 		actions done to any cell sample during an experiment. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`sample_actions_tracker` (
  `date_time` DATETIME NOT NULL,
  `sample_id` INT NOT NULL,
  `action_id` INT NULL,
  `notes` TEXT NULL,
  PRIMARY KEY (`date_time`, `sample_id`),
  INDEX `fk_sample_actions_tracker_sample_inventory1_idx` (`sample_id` ASC) VISIBLE,
  INDEX `fk_sample_actions_tracker_action_inventory1_idx` (`action_id` ASC) VISIBLE,
  CONSTRAINT `fk_sample_actions_tracker_sample_inventory1`
    FOREIGN KEY (`sample_id`)
    REFERENCES `Experiment_Automation`.`sample_inventory` (`sample_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_sample_actions_tracker_action_inventory1`
    FOREIGN KEY (`action_id`)
    REFERENCES `Experiment_Automation`.`action_inventory` (`action_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- plate_actions_tracker
-- 		"plate_actions_tracker" records all changes or actions 
-- 		done to any plate during an experiment. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`plate_actions_tracker` (
  `date_time` DATETIME NOT NULL,
  `plate_id` INT NOT NULL,
  `action_id` INT NULL,
  `notes` TEXT NULL,
  PRIMARY KEY (`date_time`, `plate_id`),
  INDEX `fk_plate_actions_tracker_action_inventory1_idx` (`action_id` ASC) VISIBLE,
  INDEX `fk_plate_actions_tracker_plate_inventory1_idx` (`plate_id` ASC) VISIBLE,
  CONSTRAINT `fk_plate_actions_tracker_action_inventory1`
    FOREIGN KEY (`action_id`)
    REFERENCES `Experiment_Automation`.`action_inventory` (`action_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_plate_actions_tracker_plate_inventory1`
    FOREIGN KEY (`plate_id`)
    REFERENCES `Experiment_Automation`.`plate_inventory` (`plate_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- well_actions_tracker 
-- 		"well_actions_tracker" records any changes or actions 
-- 		done to any well during an experiment. 
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Experiment_Automation`.`well_actions_tracker2` (
  `date_time` DATETIME NOT NULL,
  `well_id` INT NOT NULL,
  `action_id` INT NULL,
  `notes` TEXT NULL,
  PRIMARY KEY (`date_time`, `well_id`),
  INDEX `fk_well_actions_tracker_action_inventory1_idx` (`action_id` ASC) VISIBLE,
  INDEX `fk_well_actions_tracker_well_inventory1_idx` (`well_id` ASC) VISIBLE,
  CONSTRAINT `fk_well_actions_tracker2_action_inventory1`
    FOREIGN KEY (`action_id`)
    REFERENCES `Experiment_Automation`.`action_inventory` (`action_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_well_actions_tracker2_well_inventory1`
    FOREIGN KEY (`well_id`)
    REFERENCES `Experiment_Automation`.`well_inventory` (`well_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;

SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

SHOW TABLES IN Experiment_Automation;