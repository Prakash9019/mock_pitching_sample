// JavaScript logic will be filled in later// DOM Elements
        const startBtn = document.getElementById('startBtn');
        const recordBtn = document.getElementById('recordBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        const endBtn = document.getElementById('endBtn');
        const audioPlayer = document.getElementById('audioPlayer');
        const statusElement = document.getElementById('status');
        const personaSelect = document.getElementById('personaSelect');
        const transcriptDisplay = document.getElementById('transcriptDisplay');
        const conversationProgress = document.getElementById('conversationProgress');
        const investorName = document.getElementById('investorName');
        const currentStage = document.getElementById('currentStage');
        const progressFill = document.getElementById('progressFill');
        const stageCounter = document.getElementById('stageCounter');
        
        // Speech recognition variables
        let recognition;
        let isListening = false;
        let currentTranscript = '';
        let interimTranscript = '';
        let audioStream;
        let socket;
        
        // Investor persona information
        const investorPersonas = {
          'skeptical': {
            name: 'Sarah Martinez',
            title: 'Senior Partner at Venture Capital',
            description: 'Analytical and thorough investor who asks tough questions'
          },
          'technical': {
            name: 'Dr. Alex Chen', 
            title: 'CTO-turned-Investor at TechVentures',
            description: 'Tech-focused investor interested in deep technical details'
          },
          'friendly': {
            name: 'Michael Thompson',
            title: 'Angel Investor & Former Entrepreneur',
            description: 'Supportive investor focused on founder journey'
          }
        };

        // Initialize socket connection
        function initSocket() {
          socket = io({
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 1000
          });

          socket.on('connect', () => {
            console.log('Connected to server');
            updateStatus('Connected. Ready to start the pitch session.', 'info');
          });

          socket.on('disconnect', () => {
            updateStatus('Disconnected from server. Please refresh the page.', 'error');
          });

          socket.on('session_started', (data) => {
            console.log('Session started:', data);
            updateStatus(`Session started with ${data.persona} persona`, 'success');
          });

          socket.on('error', (data) => {
            console.error('Socket error:', data);
            updateStatus(`Error: ${data.message}`, 'error');
          });

          socket.on('response', (data) => {
            try {
              console.log('Received response:', data);
              
              // Check if we have an audio URL
              if (data.audio_url) {
                // Construct full URL for audio
                const audioUrl = `http://127.0.0.1:8080${data.audio_url}`;
                console.log('Playing audio from:', audioUrl);
                
                audioPlayer.src = audioUrl;
                audioPlayer.play().catch(error => {
                  console.error('Error playing audio:', error);
                  updateStatus('Error playing audio response', 'error');
                });
                
                updateStatus(`AI: ${data.message}`, 'success');
              } else {
                updateStatus(`AI: ${data.message}`, 'success');
              }
              
              // Hide the transcript display after receiving response
              setTimeout(() => {
                transcriptDisplay.style.display = 'none';
              }, 2000);
              
              // Update conversation progress
              updateConversationProgress(personaSelect.value);
            } catch (error) {
              console.error('Error handling AI response:', error);
              updateStatus('Error handling AI response', 'error');
            }
          });
        }

        // Update status message
        function updateStatus(message, type = 'info') {
          statusElement.textContent = message;
          statusElement.style.color = type === 'error' ? '#f44336' : 
                                     type === 'success' ? '#4CAF50' : '#2196F3';
        }
        
        // Update conversation progress
        function updateConversationProgress(persona) {
          const personaInfo = investorPersonas[persona];
          if (personaInfo) {
            investorName.textContent = `${personaInfo.name} - ${personaInfo.title}`;
            conversationProgress.style.display = 'block';
            
            // Fetch and update conversation stats
            if (socket && socket.id) {
              fetch(`/api/conversation/${socket.id}/stats`)
                .then(response => response.json())
                .then(stats => {
                  if (stats.topics_covered) {
                    const currentTopic = stats.topics_covered[stats.topics_covered.length - 1] || 'getting_started';
                    currentStage.textContent = currentTopic.replace('_', ' ').toUpperCase();
                    progressFill.style.width = stats.progress_percentage + '%';
                    stageCounter.textContent = `${stats.topics_covered.length} of 8 topics covered`;
                  }
                })
                .catch(error => console.log('Stats not available yet'));
            }
          }
        }

        // Start Meeting
        startBtn.addEventListener('click', async () => {
          try {
            updateStatus('Initializing meeting...', 'info');
            initSocket();
            
            // Check for speech recognition support
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
              throw new Error('Speech recognition not supported in this browser. Please use Chrome, Edge, or Safari.');
            }
            
            // Initialize speech recognition
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            
            // Configure speech recognition
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            recognition.maxAlternatives = 1;
            
            // Handle speech recognition results
            recognition.onresult = (event) => {
              let finalTranscript = '';
              let interimTranscript = '';
              
              for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                  finalTranscript += transcript + ' ';
                } else {
                  interimTranscript += transcript;
                }
              }
              
              if (finalTranscript) {
                currentTranscript += finalTranscript;
              }
              
              // Update transcript display with both final and interim results
              const displayText = currentTranscript + interimTranscript;
              if (displayText.trim()) {
                transcriptDisplay.innerHTML = `<strong>Your speech:</strong><br>"${displayText}"`;
                transcriptDisplay.style.color = '#333';
                transcriptDisplay.style.fontStyle = 'normal';
              }
              
              // Update status
              if (interimTranscript) {
                updateStatus('Listening... Keep speaking.', 'info');
              } else if (finalTranscript) {
                updateStatus('Listening... Speech captured.', 'info');
              }
            };
            
            recognition.onerror = (event) => {
              console.error('Speech recognition error:', event.error);
              updateStatus(`Speech recognition error: ${event.error}`, 'error');
            };
            
            recognition.onend = () => {
              if (isListening) {
                // Restart recognition if we're still supposed to be listening
                try {
                  recognition.start();
                } catch (error) {
                  console.error('Error restarting recognition:', error);
                }
              }
            };
            
            // Request microphone access for audio level monitoring
            try {
              audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: { echoCancellation: true, noiseSuppression: true } 
              });
              setupAudioMeter(audioStream);
            } catch (error) {
              console.warn('Could not access microphone for audio level monitoring:', error);
            }
            
            // Update UI
            startBtn.style.display = 'none';
            recordBtn.style.display = 'inline-block';
            endBtn.style.display = 'inline-block';
            updateStatus('Meeting started. Click "Start Speaking" to begin real-time speech recognition.', 'success');
            
            // Show conversation progress
            updateConversationProgress(personaSelect.value);
            
          } catch (error) {
            console.error('Error starting meeting:', error);
            updateStatus(`Error: ${error.message}`, 'error');
          }
        });

        // Start Speaking Button
        recordBtn.addEventListener('click', () => {
          if (!isListening) {
            // Start speech recognition
            try {
              currentTranscript = ''; // Reset transcript
              transcriptDisplay.style.display = 'block';
              transcriptDisplay.innerHTML = 'Listening... Start speaking now.';
              transcriptDisplay.style.color = '#666';
              transcriptDisplay.style.fontStyle = 'italic';
              
              recognition.start();
              isListening = true;
              recordBtn.classList.add('active');
              recordBtn.textContent = '🎙️ Listening...';
              pauseBtn.style.display = 'inline-block';
              document.getElementById('audio-level-container').style.display = 'block';
              updateStatus('Listening... Start speaking now.', 'info');
            } catch (error) {
              console.error('Error starting speech recognition:', error);
              updateStatus('Error starting speech recognition', 'error');
            }
          }
        });

        // Pause & Send Button
        pauseBtn.addEventListener('click', () => {
          if (isListening) {
            // Stop speech recognition and send the transcript
            recognition.stop();
            isListening = false;
            recordBtn.classList.remove('active');
            recordBtn.textContent = '🎙️ Start Speaking';
            pauseBtn.style.display = 'none';
            document.getElementById('audio-level-container').style.display = 'none';
            
            if (currentTranscript.trim()) {
              transcriptDisplay.innerHTML = `<strong>Sent:</strong><br>"${currentTranscript.trim()}"`;
              transcriptDisplay.style.color = '#2196F3';
              updateStatus('Sending your message to AI...', 'info');
              socket.emit('text_message', {
                text: currentTranscript.trim(),
                persona: personaSelect.value
              });
            } else {
              transcriptDisplay.style.display = 'none';
              updateStatus('No speech detected. Please try again.', 'error');
            }
          }
        });

        // End Session
        endBtn.addEventListener('click', () => {
          if (isListening) {
            recognition.stop();
            isListening = false;
          }
          
          // Stop all tracks in the stream
          if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
          }
          
          // Reset UI
          startBtn.style.display = 'inline-block';
          recordBtn.style.display = 'none';
          pauseBtn.style.display = 'none';
          endBtn.style.display = 'none';
          recordBtn.textContent = '🎙️ Start Speaking';
          document.getElementById('audio-level-container').style.display = 'none';
          transcriptDisplay.style.display = 'none';
          conversationProgress.style.display = 'none';
          
          updateStatus('Session ended. Click "Start Meeting" to begin a new session.', 'info');
          
          // Disconnect socket
          if (socket) {
            socket.disconnect();
          }
        });

        // Audio level monitoring function
        function setupAudioMeter(stream) {
          try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            source.connect(analyser);
            
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            
            function checkLevel() {
              analyser.getByteFrequencyData(dataArray);
              const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
              // Visual feedback for audio level
              const levelIndicator = document.getElementById('audio-level');
              if (levelIndicator) {
                levelIndicator.style.width = `${Math.min(100, average)}%`;
                levelIndicator.style.backgroundColor = average > 50 ? '#4CAF50' : '#2196F3';
              }
              requestAnimationFrame(checkLevel);
            }
            checkLevel();
          } catch (error) {
            console.warn('Audio level monitoring not available:', error);
          }
        }

        // Check for browser support
        if (!navigator.mediaDevices || !window.MediaRecorder) {
          updateStatus('Your browser does not support audio recording. Please use Chrome, Firefox, or Edge.', 'error');
          startBtn.disabled = true;
        }