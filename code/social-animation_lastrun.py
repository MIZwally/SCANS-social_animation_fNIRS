#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on November 18, 2025, at 10:48
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from trigger_test_code
from pylsl import resolve_streams, StreamOutlet, StreamInfo
info = StreamInfo(name='Trigger1', type='Markers', channel_count=1, channel_format='int32', source_id='Tangrams')  # pyright: ignore[reportArgumentType]
outlet = StreamOutlet(info)
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'social-animation'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'Dyad ID': '',
    'Subject ID': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1536, 960]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'../data/%s_%s_%s' % (expInfo['Dyad ID'], expInfo['Subject ID'], expName)
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Public\\Desktop\\SCANS Tasks\\SCANS-social_animation_fNIRS\\code\\social-animation_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='priority'
    )
    thisExp.setPriority('stimuli', priority.HIGH)
    thisExp.setPriority('category', priority.HIGH)
    thisExp.setPriority('resp_key.keys', priority.HIGH)
    thisExp.setPriority('Dyad ID', priority.CRITICAL)
    thisExp.setPriority('Subject ID', priority.CRITICAL)
    thisExp.setPriority('thisN', priority.HIGH)
    thisExp.setPriority('runs', priority.HIGH)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('trigger_test_space') is None:
        # initialise trigger_test_space
        trigger_test_space = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='trigger_test_space',
        )
    if deviceManager.getDevice('start_task') is None:
        # initialise start_task
        start_task = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start_task',
        )
    # create speaker 'start_sound'
    deviceManager.addDevice(
        deviceName='start_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('resp_key') is None:
        # initialise resp_key
        resp_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='resp_key',
        )
    # create speaker 'end_sound'
    deviceManager.addDevice(
        deviceName='end_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "trigger_test" ---
    trigger_text = visual.TextStim(win=win, name='trigger_text',
        text='Press space to send a test trigger.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    trigger_test_space = keyboard.Keyboard(deviceName='trigger_test_space')
    
    # --- Initialize components for Routine "intro" ---
    Intro = visual.TextStim(win=win, name='Intro',
        text='You will now watch short clips of shapes and decide if they are having a social interaction or not. \n\nWhen the clip is finished, you will be prompted to make a decision.\n\nFor a social interaction, press the left arrow key. For no interaction, press the right arrow key. If you are not sure, press the down arrow key. \n\n',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    start_task_text = visual.TextStim(win=win, name='start_task_text',
        text='Please wait for the experimenter to start the task.',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    start_task = keyboard.Keyboard(deviceName='start_task')
    start_sound = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='start_sound',    name='start_sound'
    )
    start_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "video_phase" ---
    video_stim = visual.MovieStim(
        win, name='video_stim',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=False,
        pos=(0, 0), size=(0.5, 0.5), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=0
    )
    
    # --- Initialize components for Routine "resp_phase" ---
    resp_prompt = visual.TextStim(win=win, name='resp_prompt',
        text='    Social             Unsure          No Interaction',
        font='Open Sans',
        pos=(0.005, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    resp_key = keyboard.Keyboard(deviceName='resp_key')
    arrow_text = visual.TextStim(win=win, name='arrow_text',
        text='(Left)              (Down)                (Right)',
        font='Arial',
        pos=(0.001, -0.05), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "fixation_phase" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "end_task" ---
    end_task_text = visual.TextStim(win=win, name='end_task_text',
        text='Thank you for participanting.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_sound = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='end_sound',    name='end_sound'
    )
    end_sound.setVolume(1.0)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "trigger_test" ---
    # create an object to store info about Routine trigger_test
    trigger_test = data.Routine(
        name='trigger_test',
        components=[trigger_text, trigger_test_space],
    )
    trigger_test.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for trigger_test_space
    trigger_test_space.keys = []
    trigger_test_space.rt = []
    _trigger_test_space_allKeys = []
    # store start times for trigger_test
    trigger_test.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    trigger_test.tStart = globalClock.getTime(format='float')
    trigger_test.status = STARTED
    thisExp.addData('trigger_test.started', trigger_test.tStart)
    trigger_test.maxDuration = None
    # keep track of which components have finished
    trigger_testComponents = trigger_test.components
    for thisComponent in trigger_test.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "trigger_test" ---
    trigger_test.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *trigger_text* updates
        
        # if trigger_text is starting this frame...
        if trigger_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            trigger_text.frameNStart = frameN  # exact frame index
            trigger_text.tStart = t  # local t and not account for scr refresh
            trigger_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(trigger_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'trigger_text.started')
            # update status
            trigger_text.status = STARTED
            trigger_text.setAutoDraw(True)
        
        # if trigger_text is active this frame...
        if trigger_text.status == STARTED:
            # update params
            pass
        
        # *trigger_test_space* updates
        waitOnFlip = False
        
        # if trigger_test_space is starting this frame...
        if trigger_test_space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            trigger_test_space.frameNStart = frameN  # exact frame index
            trigger_test_space.tStart = t  # local t and not account for scr refresh
            trigger_test_space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(trigger_test_space, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'trigger_test_space.started')
            # update status
            trigger_test_space.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(trigger_test_space.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(trigger_test_space.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if trigger_test_space.status == STARTED and not waitOnFlip:
            theseKeys = trigger_test_space.getKeys(keyList=['space', 'esc'], ignoreKeys=["escape"], waitRelease=False)
            _trigger_test_space_allKeys.extend(theseKeys)
            if len(_trigger_test_space_allKeys):
                trigger_test_space.keys = _trigger_test_space_allKeys[-1].name  # just the last key pressed
                trigger_test_space.rt = _trigger_test_space_allKeys[-1].rt
                trigger_test_space.duration = _trigger_test_space_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            trigger_test.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trigger_test.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "trigger_test" ---
    for thisComponent in trigger_test.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for trigger_test
    trigger_test.tStop = globalClock.getTime(format='float')
    trigger_test.tStopRefresh = tThisFlipGlobal
    thisExp.addData('trigger_test.stopped', trigger_test.tStop)
    # Run 'End Routine' code from trigger_test_code
    outlet.push_sample(x=[1])
    print('trigger test')
    # check responses
    if trigger_test_space.keys in ['', [], None]:  # No response was made
        trigger_test_space.keys = None
    thisExp.addData('trigger_test_space.keys',trigger_test_space.keys)
    if trigger_test_space.keys != None:  # we had a response
        thisExp.addData('trigger_test_space.rt', trigger_test_space.rt)
        thisExp.addData('trigger_test_space.duration', trigger_test_space.duration)
    thisExp.nextEntry()
    # the Routine "trigger_test" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro" ---
    # create an object to store info about Routine intro
    intro = data.Routine(
        name='intro',
        components=[Intro, start_task_text, start_task, start_sound],
    )
    intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from start_trigger
    outlet.push_sample(x=[99])
    print(99)
    # create starting attributes for start_task
    start_task.keys = []
    start_task.rt = []
    _start_task_allKeys = []
    start_sound.setSound('D', secs=0.5, hamming=True)
    start_sound.setVolume(1.0, log=False)
    start_sound.seek(0)
    # store start times for intro
    intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro.tStart = globalClock.getTime(format='float')
    intro.status = STARTED
    thisExp.addData('intro.started', intro.tStart)
    intro.maxDuration = None
    # keep track of which components have finished
    introComponents = intro.components
    for thisComponent in intro.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro" ---
    intro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Intro* updates
        
        # if Intro is starting this frame...
        if Intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Intro.frameNStart = frameN  # exact frame index
            Intro.tStart = t  # local t and not account for scr refresh
            Intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Intro, 'tStartRefresh')  # time at next scr refresh
            # update status
            Intro.status = STARTED
            Intro.setAutoDraw(True)
        
        # if Intro is active this frame...
        if Intro.status == STARTED:
            # update params
            pass
        
        # *start_task_text* updates
        
        # if start_task_text is starting this frame...
        if start_task_text.status == NOT_STARTED and tThisFlip >= 7.5-frameTolerance:
            # keep track of start time/frame for later
            start_task_text.frameNStart = frameN  # exact frame index
            start_task_text.tStart = t  # local t and not account for scr refresh
            start_task_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_task_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            start_task_text.status = STARTED
            start_task_text.setAutoDraw(True)
        
        # if start_task_text is active this frame...
        if start_task_text.status == STARTED:
            # update params
            pass
        
        # *start_task* updates
        waitOnFlip = False
        
        # if start_task is starting this frame...
        if start_task.status == NOT_STARTED and tThisFlip >= 7.5-frameTolerance:
            # keep track of start time/frame for later
            start_task.frameNStart = frameN  # exact frame index
            start_task.tStart = t  # local t and not account for scr refresh
            start_task.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_task, 'tStartRefresh')  # time at next scr refresh
            # update status
            start_task.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_task.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_task.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_task.status == STARTED and not waitOnFlip:
            theseKeys = start_task.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _start_task_allKeys.extend(theseKeys)
            if len(_start_task_allKeys):
                start_task.keys = _start_task_allKeys[-1].name  # just the last key pressed
                start_task.rt = _start_task_allKeys[-1].rt
                start_task.duration = _start_task_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *start_sound* updates
        
        # if start_sound is starting this frame...
        if start_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_sound.frameNStart = frameN  # exact frame index
            start_sound.tStart = t  # local t and not account for scr refresh
            start_sound.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            start_sound.status = STARTED
            start_sound.play(when=win)  # sync with win flip
        
        # if start_sound is stopping this frame...
        if start_sound.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > start_sound.tStartRefresh + 0.5-frameTolerance or start_sound.isFinished:
                # keep track of stop time/frame for later
                start_sound.tStop = t  # not accounting for scr refresh
                start_sound.tStopRefresh = tThisFlipGlobal  # on global time
                start_sound.frameNStop = frameN  # exact frame index
                # update status
                start_sound.status = FINISHED
                start_sound.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[start_sound]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            intro.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro" ---
    for thisComponent in intro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro
    intro.tStop = globalClock.getTime(format='float')
    intro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro.stopped', intro.tStop)
    start_sound.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Runs = data.TrialHandler2(
        name='Runs',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('../stimuli/run_files.csv'), 
        seed=None, 
    )
    thisExp.addLoop(Runs)  # add the loop to the experiment
    thisRun = Runs.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
    if thisRun != None:
        for paramName in thisRun:
            globals()[paramName] = thisRun[paramName]
    
    for thisRun in Runs:
        currentLoop = Runs
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
        if thisRun != None:
            for paramName in thisRun:
                globals()[paramName] = thisRun[paramName]
        
        # set up handler to look after randomisation of conditions etc
        Trials = data.TrialHandler2(
            name='Trials',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(file), 
            seed=None, 
        )
        thisExp.addLoop(Trials)  # add the loop to the experiment
        thisTrial = Trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in Trials:
            currentLoop = Trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "video_phase" ---
            # create an object to store info about Routine video_phase
            video_phase = data.Routine(
                name='video_phase',
                components=[video_stim],
            )
            video_phase.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            video_stim.setMovie(stimuli)
            # Run 'Begin Routine' code from routine_trigger
            i = int(Trials.thisN)
            condition = 0
            if Trials.trialList[i]['category'] == 'nonsocial' :
                condition = 1
            else :
                condition = 2
            trigger = ((i+1)*10) + condition
            outlet.push_sample(x=[trigger])
            print('trigger:', trigger)
            # store start times for video_phase
            video_phase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            video_phase.tStart = globalClock.getTime(format='float')
            video_phase.status = STARTED
            thisExp.addData('video_phase.started', video_phase.tStart)
            video_phase.maxDuration = None
            # keep track of which components have finished
            video_phaseComponents = video_phase.components
            for thisComponent in video_phase.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "video_phase" ---
            # if trial has changed, end Routine now
            if isinstance(Trials, data.TrialHandler2) and thisTrial.thisN != Trials.thisTrial.thisN:
                continueRoutine = False
            video_phase.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 20.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *video_stim* updates
                
                # if video_stim is starting this frame...
                if video_stim.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    video_stim.frameNStart = frameN  # exact frame index
                    video_stim.tStart = t  # local t and not account for scr refresh
                    video_stim.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(video_stim, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    video_stim.status = STARTED
                    video_stim.setAutoDraw(True)
                    video_stim.play()
                
                # if video_stim is stopping this frame...
                if video_stim.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > video_stim.tStartRefresh + 20-frameTolerance or video_stim.isFinished:
                        # keep track of stop time/frame for later
                        video_stim.tStop = t  # not accounting for scr refresh
                        video_stim.tStopRefresh = tThisFlipGlobal  # on global time
                        video_stim.frameNStop = frameN  # exact frame index
                        # update status
                        video_stim.status = FINISHED
                        video_stim.setAutoDraw(False)
                        video_stim.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[video_stim]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    video_phase.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in video_phase.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "video_phase" ---
            for thisComponent in video_phase.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for video_phase
            video_phase.tStop = globalClock.getTime(format='float')
            video_phase.tStopRefresh = tThisFlipGlobal
            thisExp.addData('video_phase.stopped', video_phase.tStop)
            video_stim.stop()  # ensure movie has stopped at end of Routine
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if video_phase.maxDurationReached:
                routineTimer.addTime(-video_phase.maxDuration)
            elif video_phase.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-20.000000)
            
            # --- Prepare to start Routine "resp_phase" ---
            # create an object to store info about Routine resp_phase
            resp_phase = data.Routine(
                name='resp_phase',
                components=[resp_prompt, resp_key, arrow_text],
            )
            resp_phase.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for resp_key
            resp_key.keys = []
            resp_key.rt = []
            _resp_key_allKeys = []
            # Run 'Begin Routine' code from response_trigger
            outlet.push_sample(x=[77])
            print('trigger: 77')
            # store start times for resp_phase
            resp_phase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            resp_phase.tStart = globalClock.getTime(format='float')
            resp_phase.status = STARTED
            thisExp.addData('resp_phase.started', resp_phase.tStart)
            resp_phase.maxDuration = None
            # keep track of which components have finished
            resp_phaseComponents = resp_phase.components
            for thisComponent in resp_phase.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "resp_phase" ---
            # if trial has changed, end Routine now
            if isinstance(Trials, data.TrialHandler2) and thisTrial.thisN != Trials.thisTrial.thisN:
                continueRoutine = False
            resp_phase.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 10.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *resp_prompt* updates
                
                # if resp_prompt is starting this frame...
                if resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    resp_prompt.frameNStart = frameN  # exact frame index
                    resp_prompt.tStart = t  # local t and not account for scr refresh
                    resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(resp_prompt, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    resp_prompt.status = STARTED
                    resp_prompt.setAutoDraw(True)
                
                # if resp_prompt is active this frame...
                if resp_prompt.status == STARTED:
                    # update params
                    pass
                
                # if resp_prompt is stopping this frame...
                if resp_prompt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > resp_prompt.tStartRefresh + 10-frameTolerance:
                        # keep track of stop time/frame for later
                        resp_prompt.tStop = t  # not accounting for scr refresh
                        resp_prompt.tStopRefresh = tThisFlipGlobal  # on global time
                        resp_prompt.frameNStop = frameN  # exact frame index
                        # update status
                        resp_prompt.status = FINISHED
                        resp_prompt.setAutoDraw(False)
                
                # *resp_key* updates
                waitOnFlip = False
                
                # if resp_key is starting this frame...
                if resp_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    resp_key.frameNStart = frameN  # exact frame index
                    resp_key.tStart = t  # local t and not account for scr refresh
                    resp_key.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(resp_key, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    resp_key.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(resp_key.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(resp_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if resp_key is stopping this frame...
                if resp_key.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > resp_key.tStartRefresh + 10-frameTolerance:
                        # keep track of stop time/frame for later
                        resp_key.tStop = t  # not accounting for scr refresh
                        resp_key.tStopRefresh = tThisFlipGlobal  # on global time
                        resp_key.frameNStop = frameN  # exact frame index
                        # update status
                        resp_key.status = FINISHED
                        resp_key.status = FINISHED
                if resp_key.status == STARTED and not waitOnFlip:
                    theseKeys = resp_key.getKeys(keyList=['left','down','right'], ignoreKeys=["escape"], waitRelease=False)
                    _resp_key_allKeys.extend(theseKeys)
                    if len(_resp_key_allKeys):
                        resp_key.keys = _resp_key_allKeys[-1].name  # just the last key pressed
                        resp_key.rt = _resp_key_allKeys[-1].rt
                        resp_key.duration = _resp_key_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *arrow_text* updates
                
                # if arrow_text is starting this frame...
                if arrow_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    arrow_text.frameNStart = frameN  # exact frame index
                    arrow_text.tStart = t  # local t and not account for scr refresh
                    arrow_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(arrow_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'arrow_text.started')
                    # update status
                    arrow_text.status = STARTED
                    arrow_text.setAutoDraw(True)
                
                # if arrow_text is active this frame...
                if arrow_text.status == STARTED:
                    # update params
                    pass
                
                # if arrow_text is stopping this frame...
                if arrow_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > arrow_text.tStartRefresh + 10-frameTolerance:
                        # keep track of stop time/frame for later
                        arrow_text.tStop = t  # not accounting for scr refresh
                        arrow_text.tStopRefresh = tThisFlipGlobal  # on global time
                        arrow_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'arrow_text.stopped')
                        # update status
                        arrow_text.status = FINISHED
                        arrow_text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    resp_phase.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in resp_phase.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "resp_phase" ---
            for thisComponent in resp_phase.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for resp_phase
            resp_phase.tStop = globalClock.getTime(format='float')
            resp_phase.tStopRefresh = tThisFlipGlobal
            thisExp.addData('resp_phase.stopped', resp_phase.tStop)
            # check responses
            if resp_key.keys in ['', [], None]:  # No response was made
                resp_key.keys = None
            Trials.addData('resp_key.keys',resp_key.keys)
            if resp_key.keys != None:  # we had a response
                Trials.addData('resp_key.rt', resp_key.rt)
                Trials.addData('resp_key.duration', resp_key.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if resp_phase.maxDurationReached:
                routineTimer.addTime(-resp_phase.maxDuration)
            elif resp_phase.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-10.000000)
            
            # --- Prepare to start Routine "fixation_phase" ---
            # create an object to store info about Routine fixation_phase
            fixation_phase = data.Routine(
                name='fixation_phase',
                components=[fixation],
            )
            fixation_phase.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from fixation_trigger
            outlet.push_sample(x=[66])
            print('trigger: 66')
            # store start times for fixation_phase
            fixation_phase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fixation_phase.tStart = globalClock.getTime(format='float')
            fixation_phase.status = STARTED
            thisExp.addData('fixation_phase.started', fixation_phase.tStart)
            fixation_phase.maxDuration = None
            # keep track of which components have finished
            fixation_phaseComponents = fixation_phase.components
            for thisComponent in fixation_phase.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation_phase" ---
            # if trial has changed, end Routine now
            if isinstance(Trials, data.TrialHandler2) and thisTrial.thisN != Trials.thisTrial.thisN:
                continueRoutine = False
            fixation_phase.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 5.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixation* updates
                
                # if fixation is starting this frame...
                if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation.frameNStart = frameN  # exact frame index
                    fixation.tStart = t  # local t and not account for scr refresh
                    fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    fixation.status = STARTED
                    fixation.setAutoDraw(True)
                
                # if fixation is active this frame...
                if fixation.status == STARTED:
                    # update params
                    pass
                
                # if fixation is stopping this frame...
                if fixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixation.tStartRefresh + 5-frameTolerance:
                        # keep track of stop time/frame for later
                        fixation.tStop = t  # not accounting for scr refresh
                        fixation.tStopRefresh = tThisFlipGlobal  # on global time
                        fixation.frameNStop = frameN  # exact frame index
                        # update status
                        fixation.status = FINISHED
                        fixation.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    fixation_phase.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation_phase.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation_phase" ---
            for thisComponent in fixation_phase.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fixation_phase
            fixation_phase.tStop = globalClock.getTime(format='float')
            fixation_phase.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fixation_phase.stopped', fixation_phase.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if fixation_phase.maxDurationReached:
                routineTimer.addTime(-fixation_phase.maxDuration)
            elif fixation_phase.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-5.000000)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'Trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'Runs'
    
    
    # --- Prepare to start Routine "end_task" ---
    # create an object to store info about Routine end_task
    end_task = data.Routine(
        name='end_task',
        components=[end_task_text, end_sound],
    )
    end_task.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    end_sound.setSound('C', secs=0.5, hamming=True)
    end_sound.setVolume(1.0, log=False)
    end_sound.seek(0)
    # store start times for end_task
    end_task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_task.tStart = globalClock.getTime(format='float')
    end_task.status = STARTED
    thisExp.addData('end_task.started', end_task.tStart)
    end_task.maxDuration = None
    # keep track of which components have finished
    end_taskComponents = end_task.components
    for thisComponent in end_task.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_task" ---
    end_task.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_task_text* updates
        
        # if end_task_text is starting this frame...
        if end_task_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_task_text.frameNStart = frameN  # exact frame index
            end_task_text.tStart = t  # local t and not account for scr refresh
            end_task_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_task_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            end_task_text.status = STARTED
            end_task_text.setAutoDraw(True)
        
        # if end_task_text is active this frame...
        if end_task_text.status == STARTED:
            # update params
            pass
        
        # if end_task_text is stopping this frame...
        if end_task_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_task_text.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                end_task_text.tStop = t  # not accounting for scr refresh
                end_task_text.tStopRefresh = tThisFlipGlobal  # on global time
                end_task_text.frameNStop = frameN  # exact frame index
                # update status
                end_task_text.status = FINISHED
                end_task_text.setAutoDraw(False)
        
        # *end_sound* updates
        
        # if end_sound is starting this frame...
        if end_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_sound.frameNStart = frameN  # exact frame index
            end_sound.tStart = t  # local t and not account for scr refresh
            end_sound.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('end_sound.started', tThisFlipGlobal)
            # update status
            end_sound.status = STARTED
            end_sound.play(when=win)  # sync with win flip
        
        # if end_sound is stopping this frame...
        if end_sound.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_sound.tStartRefresh + 0.5-frameTolerance or end_sound.isFinished:
                # keep track of stop time/frame for later
                end_sound.tStop = t  # not accounting for scr refresh
                end_sound.tStopRefresh = tThisFlipGlobal  # on global time
                end_sound.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_sound.stopped')
                # update status
                end_sound.status = FINISHED
                end_sound.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[end_sound]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end_task.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_task.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_task" ---
    for thisComponent in end_task.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_task
    end_task.tStop = globalClock.getTime(format='float')
    end_task.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_task.stopped', end_task.tStop)
    # Run 'End Routine' code from end_trigger
    outlet.push_sample(x=[99])
    print(99)
    end_sound.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end_task.maxDurationReached:
        routineTimer.addTime(-end_task.maxDuration)
    elif end_task.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
