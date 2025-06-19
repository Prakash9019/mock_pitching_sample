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
        let currentSessionId = null;
        let sessionStartTime = null;
        
        // Investor persona information (will be loaded from API)
        let investorPersonas = {};

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
                // Use relative URL to avoid hardcoding host/port
                const audioUrl = data.audio_url;
                console.log('Playing audio from:', audioUrl);
                
                // Force reload by adding timestamp to prevent caching
                const cacheBuster = `?t=${Date.now()}`;
                audioPlayer.src = audioUrl + cacheBuster;
                audioPlayer.load(); // Force reload of audio element
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
            
            // Generate session ID and track start time
            currentSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            sessionStartTime = new Date();
            
            // Update UI
            startBtn.style.display = 'none';
            recordBtn.style.display = 'inline-block';
            endBtn.style.display = 'inline-block';
            updateStatus('Meeting started. Click "Start Speaking" to begin real-time speech recognition.', 'success');
            
            // Show conversation progress and analysis buttons
            updateConversationProgress(personaSelect.value);
            showAnalysisButtons();
            
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
                persona: personaSelect.value,
                session_id: currentSessionId,
                system: 'workflow'
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

        // ===== PERSONA LOADING FUNCTIONS =====

        // Load personas from API
        async function loadPersonas() {
          try {
            const response = await fetch('/api/personas');
            const data = await response.json();
            
            if (data.success && data.personas) {
              investorPersonas = data.personas;
              updatePersonaSelect();
              console.log('Personas loaded successfully:', Object.keys(investorPersonas));
            } else {
              throw new Error('Failed to load personas');
            }
          } catch (error) {
            console.error('Error loading personas:', error);
            // Fallback to default personas
            investorPersonas = {
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
            updatePersonaSelect();
          }
        }

        // Update persona select dropdown
        function updatePersonaSelect() {
          const personaSelect = document.getElementById('personaSelect');
          if (!personaSelect) return;
          
          // Clear existing options
          personaSelect.innerHTML = '';
          
          // Add personas from API
          Object.keys(investorPersonas).forEach(personaKey => {
            const persona = investorPersonas[personaKey];
            const option = document.createElement('option');
            option.value = personaKey;
            option.textContent = `${getPersonaEmoji(personaKey)} ${persona.name}`;
            option.title = persona.description;
            
            // Set default selection
            if (personaKey === 'skeptical') {
              option.selected = true;
            }
            
            personaSelect.appendChild(option);
          });
          
          // Add change event listener
          personaSelect.addEventListener('change', updatePersonaInfo);
          
          // Update persona info for initial selection
          updatePersonaInfo();
        }

        // Update persona info display
        function updatePersonaInfo() {
          const personaSelect = document.getElementById('personaSelect');
          const selectedPersona = personaSelect?.value;
          
          if (!selectedPersona || !investorPersonas[selectedPersona]) return;
          
          const persona = investorPersonas[selectedPersona];
          const personaInfo = document.getElementById('personaInfo');
          const personaAvatar = document.getElementById('personaAvatar');
          const personaName = document.getElementById('personaName');
          const personaTitle = document.getElementById('personaTitle');
          const personaDescription = document.getElementById('personaDescription');
          const personaTraits = document.getElementById('personaTraits');
          
          if (personaInfo) {
            personaInfo.classList.remove('hidden');
            
            if (personaAvatar) personaAvatar.textContent = getPersonaEmoji(selectedPersona);
            if (personaName) personaName.textContent = persona.name;
            if (personaTitle) personaTitle.textContent = persona.title;
            if (personaDescription) personaDescription.textContent = persona.description;
            
            // Update traits
            if (personaTraits && persona.personality_traits) {
              personaTraits.innerHTML = '';
              persona.personality_traits.forEach(trait => {
                const span = document.createElement('span');
                span.className = 'bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full';
                span.textContent = trait;
                personaTraits.appendChild(span);
              });
            }
          }
        }

        // Get emoji for persona
        function getPersonaEmoji(personaKey) {
          const emojiMap = {
            'friendly': '🤝',
            'skeptical': '🧐',
            'technical': '🧠'
          };
          return emojiMap[personaKey] || '👤';
        }

        // ===== PITCH ANALYSIS FUNCTIONS =====

        // Show analysis buttons
        function showAnalysisButtons() {
          const quickAnalysisBtn = document.getElementById('quickAnalysisBtn');
          const fullAnalysisBtn = document.getElementById('fullAnalysisBtn');
          if (quickAnalysisBtn) quickAnalysisBtn.classList.remove('hidden');
          if (fullAnalysisBtn) fullAnalysisBtn.classList.remove('hidden');
        }

        // Hide analysis buttons
        function hideAnalysisButtons() {
          const quickAnalysisBtn = document.getElementById('quickAnalysisBtn');
          const fullAnalysisBtn = document.getElementById('fullAnalysisBtn');
          if (quickAnalysisBtn) quickAnalysisBtn.classList.add('hidden');
          if (fullAnalysisBtn) fullAnalysisBtn.classList.add('hidden');
        }

        // Update real-time analysis display
        function updateRealtimeAnalysis(analysis) {
          const realtimeAnalysis = document.getElementById('realtimeAnalysis');
          const currentScore = document.getElementById('currentScore');
          const keyInsights = document.getElementById('keyInsights');
          
          if (!realtimeAnalysis || !currentScore || !keyInsights) return;
          
          realtimeAnalysis.classList.remove('hidden');
          currentScore.textContent = `${analysis.overall_score || 0}/100`;
          
          // Update key insights
          keyInsights.innerHTML = '';
          const insights = analysis.key_recommendations || analysis.strengths || [];
          if (insights.length > 0) {
            insights.slice(0, 3).forEach(insight => {
              const li = document.createElement('li');
              li.textContent = typeof insight === 'string' ? insight : insight.description || insight.area || 'Insight available';
              li.className = 'text-sm text-gray-600 mb-1';
              keyInsights.appendChild(li);
            });
          } else {
            const li = document.createElement('li');
            li.textContent = 'Keep speaking to generate insights...';
            li.className = 'text-sm text-gray-500 italic';
            keyInsights.appendChild(li);
          }
        }

        // Get quick analysis
        async function getQuickAnalysis() {
          if (!currentSessionId) {
            updateStatus('No active session for analysis', 'error');
            return;
          }

          try {
            updateStatus('Generating quick analysis...', 'info');
            const response = await fetch(`/api/pitch/analytics/${currentSessionId}`);
            const data = await response.json();
            
            if (data.success && data.analytics) {
              // Create a simplified analysis for real-time display
              const quickAnalysis = {
                overall_score: Math.min(100, (data.analytics.total_questions * 10) + (Object.keys(data.analytics.key_insights).length * 5)),
                key_recommendations: [
                  `Completed ${data.analytics.completed_stages.length}/9 stages`,
                  `Asked ${data.analytics.total_questions} questions`,
                  `Generated ${Object.values(data.analytics.key_insights).flat().length} insights`
                ]
              };
              updateRealtimeAnalysis(quickAnalysis);
              updateStatus('Quick analysis updated', 'success');
            } else {
              throw new Error(data.error || 'Failed to get analytics');
            }
          } catch (error) {
            console.error('Error getting quick analysis:', error);
            updateStatus('Error getting quick analysis', 'error');
          }
        }

        // End session with analysis
        async function endSessionWithAnalysis() {
          if (!currentSessionId) {
            updateStatus('No active session to end', 'error');
            return;
          }

          try {
            updateStatus('Ending session and generating analysis...', 'info');
            
            const response = await fetch(`/api/pitch/end/${currentSessionId}`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ reason: 'user_ended' })
            });
            
            const data = await response.json();
            
            if (data.success && data.analysis) {
              // Show analysis modal
              showAnalysisModal(data.analysis);
              updateStatus('Session ended. Analysis generated!', 'success');
            } else {
              throw new Error(data.error || 'Failed to generate analysis');
            }
          } catch (error) {
            console.error('Error ending session with analysis:', error);
            updateStatus('Error generating analysis', 'error');
          }
        }

        // Show analysis modal
        function showAnalysisModal(analysis) {
          // Create modal HTML
          const modalHTML = `
            <div id="analysisModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
              <div class="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
                <div class="p-6">
                  <!-- Header -->
                  <div class="flex justify-between items-center mb-6">
                    <h2 class="text-2xl font-bold text-gray-900">🎯 Pitch Analysis Report</h2>
                    <button onclick="closeAnalysisModal()" class="text-gray-500 hover:text-gray-700 text-2xl">&times;</button>
                  </div>
                  
                  <!-- Session Info -->
                  <div class="bg-gray-50 rounded-lg p-4 mb-6">
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                      <div>
                        <div class="text-2xl font-bold text-blue-600">${analysis.overall_score || 0}</div>
                        <div class="text-sm text-gray-600">Overall Score</div>
                      </div>
                      <div>
                        <div class="text-2xl font-bold text-green-600">${analysis.completion_percentage || 0}%</div>
                        <div class="text-sm text-gray-600">Completed</div>
                      </div>
                      <div>
                        <div class="text-2xl font-bold text-purple-600">${analysis.session_duration_minutes || 0}m</div>
                        <div class="text-sm text-gray-600">Duration</div>
                      </div>
                      <div>
                        <div class="text-lg font-bold text-indigo-600">${analysis.pitch_readiness || 'Unknown'}</div>
                        <div class="text-sm text-gray-600">Readiness</div>
                      </div>
                    </div>
                  </div>
                  
                  <!-- Strengths and Weaknesses -->
                  <div class="grid md:grid-cols-2 gap-6 mb-6">
                    <!-- Strengths -->
                    <div class="bg-green-50 rounded-lg p-4">
                      <h3 class="text-lg font-semibold text-green-800 mb-3">✅ Strengths</h3>
                      <div class="space-y-2">
                        ${(analysis.strengths || []).slice(0, 3).map(strength => `
                          <div class="flex items-start">
                            <span class="text-green-500 mr-2">•</span>
                            <div>
                              <div class="font-medium text-green-700">${strength.area || 'Strength'}</div>
                              <div class="text-sm text-green-600">${strength.description || ''}</div>
                            </div>
                          </div>
                        `).join('')}
                      </div>
                    </div>
                    
                    <!-- Weaknesses -->
                    <div class="bg-red-50 rounded-lg p-4">
                      <h3 class="text-lg font-semibold text-red-800 mb-3">❌ Areas for Improvement</h3>
                      <div class="space-y-2">
                        ${(analysis.weaknesses || []).slice(0, 3).map(weakness => `
                          <div class="flex items-start">
                            <span class="text-red-500 mr-2">•</span>
                            <div>
                              <div class="font-medium text-red-700">${weakness.area || 'Area'}</div>
                              <div class="text-sm text-red-600">${weakness.description || ''}</div>
                            </div>
                          </div>
                        `).join('')}
                      </div>
                    </div>
                  </div>
                  
                  <!-- Recommendations -->
                  <div class="bg-blue-50 rounded-lg p-4 mb-6">
                    <h3 class="text-lg font-semibold text-blue-800 mb-3">🎯 Key Recommendations</h3>
                    <ul class="space-y-2">
                      ${(analysis.key_recommendations || []).slice(0, 5).map(rec => `
                        <li class="flex items-start">
                          <span class="text-blue-500 mr-2">→</span>
                          <span class="text-blue-700">${rec}</span>
                        </li>
                      `).join('')}
                    </ul>
                  </div>
                  
                  <!-- Actions -->
                  <div class="flex flex-wrap gap-3 justify-center">
                    <button onclick="viewFullAnalysis()" class="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-lg">
                      📈 View Full Report
                    </button>
                    <button onclick="startNewSession()" class="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg">
                      🚀 Start New Session
                    </button>
                    <button onclick="closeAnalysisModal()" class="bg-gray-600 hover:bg-gray-700 text-white px-6 py-2 rounded-lg">
                      Close
                    </button>
                  </div>
                </div>
              </div>
            </div>
          `;
          
          // Add modal to page
          document.body.insertAdjacentHTML('beforeend', modalHTML);
        }

        // Global functions for modal actions
        window.closeAnalysisModal = function() {
          const modal = document.getElementById('analysisModal');
          if (modal) {
            modal.remove();
          }
        }

        window.viewFullAnalysis = function() {
          if (currentSessionId) {
            window.open(`/pitch-analysis/${currentSessionId}`, '_blank');
          }
        }

        window.startNewSession = function() {
          closeAnalysisModal();
          location.reload();
        }

        // Event Listeners for Analysis Buttons
        document.addEventListener('DOMContentLoaded', function() {
          // Load personas from API
          loadPersonas();
          
          // Quick Analysis Button
          const quickAnalysisBtn = document.getElementById('quickAnalysisBtn');
          if (quickAnalysisBtn) {
            quickAnalysisBtn.addEventListener('click', getQuickAnalysis);
          }

          // Full Analysis Button
          const fullAnalysisBtn = document.getElementById('fullAnalysisBtn');
          if (fullAnalysisBtn) {
            fullAnalysisBtn.addEventListener('click', function(e) {
              e.preventDefault();
              if (currentSessionId) {
                window.open(`/pitch-analysis/${currentSessionId}`, '_blank');
              } else {
                updateStatus('No active session for analysis', 'error');
              }
            });
          }
        });