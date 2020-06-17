let audioContext = new AudioContext();
let resonanceAudioScene = new ResonanceAudio(audioContext);
resonanceAudioScene.output.connect(audioContext.destination);

let roomDimensions = {
  width: 3.1,
  height: 2.5,
  depth: 3.4,
};

let roomMaterials = {
  // Room wall materials
  left: 'brick-bare',
  right: 'curtain-heavy',
  front: 'marble',
  back: 'glass-thin',
  // Room floor
  down: 'grass',
  // Room ceiling
  up: 'transparent',
};

resonanceAudioScene.setRoomProperties(roomDimensions, roomMaterials);

let audioElement = document.createElement('audio');
audioElement.src = 'ice.wav';

let audioElementSource = audioContext.createMediaElementSource(audioElement);

let source = resonanceAudioScene.createSource();
audioElementSource.connect(source.input);
source.setPosition(-0.707, -0.707, 0);
audioElement.play();