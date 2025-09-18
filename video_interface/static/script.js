// =====================
// Dynamic Style Section
// =====================

// Set profile panel height to match video container
const video = document.getElementById("video_container");
const profile = document.getElementById("profile");
profile.style.height = `${video.offsetHeight}px`;

// =====================
// UI Initialization
// =====================

// Hide logs panel when close button is clicked
document.getElementById('closeLogsBtn').addEventListener('click', () => {
    document.getElementById('logs').style.display = 'none';
});

// Disable logs button by default
document.getElementById('show_logs_button').disabled = true;

// Set initial user label text
document.getElementById('userLabel').innerText = 'Select User';

// =====================
// Global Constants
// =====================

const chatbox = document.getElementById("chatbox");
const live_profile = document.getElementById("live_profile");
const user_label = document.getElementById("user_label");

// =====================
// Global Variables
// =====================

let userId = ""; // Default user ID

// ====================
// ===  FUNCTIONS   ===
// ====================

// ====================
// Utility Functions
// ====================

// Function to add a message to the chatbox
function addMessage(sender, text, role) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = `${sender.trim()}: ${text.trim()}`;
    chatbox.appendChild(div);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function populateTableFromProfile(userProfile) {
    const table = document.getElementById("edit_profile");
    const thead = table.querySelector("thead");
    const tbody = table.querySelector("tbody");

    // Clear existing content
    thead.innerHTML = "";
    tbody.innerHTML = "";

    if (userProfile.length === 0) return;

    // Create table header from keys
    const headerRow = document.createElement("tr");
    const keys = Object.keys(userProfile[0]);
    keys.forEach(key => {
        if (key !== 'type') {
            const th = document.createElement("th");
            th.innerText = key;
            headerRow.appendChild(th);
        }
    });
    thead.appendChild(headerRow);

    // Create table rows
    userProfile.forEach(item => {
        const row = document.createElement("tr");
        // if type is primary, add a particular id to the row
        if (item.type === "primary") {
            row.classList.add("primary-feature");
        }
        keys.forEach(key => {
            if (key !== 'type') {
                const td = document.createElement("td");
                td.innerText = item[key];
                row.appendChild(td);
            }
        });
        tbody.appendChild(row);
    });
}

// ====================
// Video handling
// ====================

function handleVideoSelect(event) {
    const file = event.target.files[0];
    if (file) {
        prepareVideoUIForProcessing();
        previewVideoFile(file);
        processVideoFile(file);
    } else {
        resetVideoPreviewUI();
        addMessage("Change video", "Aucune vidéo sélectionnée", "system");
    }
}

function prepareVideoUIForProcessing() {
    const videoInput = document.getElementById('videoInput');
    const videoLabel = document.querySelector('#change_video label');
    document.getElementById('show_logs_button').disabled = true;
    videoInput.disabled = true;
    videoLabel.classList.add('disabled');
    document.getElementById('processingOverlay').style.visibility = 'visible';
}

function previewVideoFile(file) {
    const preview = document.getElementById('video_preview');
    const videoPlaceholder = document.getElementById('video_placeholder');
    const reader = new FileReader();
    reader.onload = function(e) {
        preview.src = e.target.result;
        preview.style.display = 'inline';
        videoPlaceholder.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function processVideoFile(file) {
    const formData = new FormData();
    formData.append('video', file);
    fetch('/set_video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('processingOverlay').style.visibility = 'hidden';
        handleVideoProcessingResponse(data);
    });
}

function handleVideoProcessingResponse(data) {
    addMessage(data.detected_user, data.transcript, "user");
    updateUserLabelAndImage(data);
    updateLiveProfile(data.profile);

    showWritingIndicator();

    // Prepare and send answer request
    const formDataAnswer = new FormData();
    formDataAnswer.append('emotion', data.emotion);
    formDataAnswer.append('current_user', data.detected_user);
    formDataAnswer.append('question', data.transcript);

    fetch('/answer', {
        method: 'POST',
        body: formDataAnswer
    })
    .then(response => response.json())
    .then(answerData => {
        removeWritingIndicator();
        if (answerData.answer) {
            addMessage("Assistant", answerData.answer, "bot");
        } else {
            addMessage("Assistant", "Aucune réponse disponible pour cette question.", "system");
        }
        updateLiveProfile(answerData.profile);
        restoreVideoUIAfterProcessing(answerData.logs);
    });
}

function updateUserLabelAndImage(data) {
    user_label.textContent = data.detected_user;
    const userPreview = document.getElementById('user_preview');
    if (data.user_image) {
        userPreview.src = "data:image/png;base64," + data.user_image;
        userPreview.style.display = 'inline';
        document.getElementById('user_placeholder').style.display = 'none';
    } else {
        userPreview.style.display = 'none';
        document.getElementById('user_placeholder').style.display = 'inline';
    }
}

function updateLiveProfile(profile) {
    if (!profile) {
        live_profile.innerHTML = "";
        return;
    }
    const primary = profile.filter(item => item.type === "primary");
    const contextual = profile.filter(item => item.type === "contextual");
    let html = "";
    if (primary.length > 0) {
        html += primary.map(
            item => `<b>${item.name}:</b> <span style="color:gray"><em>(${item.description})</em></span> </br>${item.value}`
        ).join('<br>');
    }
    if (contextual.length > 0) {
        if (primary.length > 0) html += "<br>----------<br>";
        html += contextual.map(
            item => `<b>${item.name}:</b> <span style="color:gray"><em>(${item.description})</em></span> </br>${item.value}`
        ).join('<br>');
    }
    live_profile.innerHTML = html;
}

function showWritingIndicator() {
    const writingIndicator = document.createElement("div");
    writingIndicator.className = "writing";
    writingIndicator.innerHTML = `
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
    `;
    chatbox.appendChild(writingIndicator);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function removeWritingIndicator() {
    const existingWriting = document.querySelector('.writing');
    if (existingWriting) {
        existingWriting.remove();
    }
}

function restoreVideoUIAfterProcessing(logs) {
    const videoInput = document.getElementById('videoInput');
    const videoLabel = document.querySelector('#change_video label');
    videoInput.disabled = false;
    videoLabel.classList.remove('disabled');
    const logContent = document.getElementById("log_content");
    logContent.innerHTML = logs || "Nothing to show yet";
    document.getElementById('show_logs_button').disabled = false;
}

function resetVideoPreviewUI() {
    const preview = document.getElementById('video_preview');
    const videoPlaceholder = document.getElementById('video_placeholder');
    preview.src = '';
    preview.style.display = 'none';
    videoPlaceholder.style.display = 'inline';
}

// ====================
// Show logs
// ====================

function showLogs() {
    const logs = document.getElementById("logs");
    if (logs.style.display === "none" || logs.style.display === "") {
        logs.style.display = "block";
    } else {
        logs.style.display = "none";
    }
}

// ====================
// Context Management
// ====================

function resetSession() {
    fetch('/reset_session', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            chatbox.innerHTML = "";
            live_profile.innerHTML = "";
            user_label.textContent = "";
            document.getElementById('video_preview').src = "";
            document.getElementById('video_preview').style.display = 'none';
            document.getElementById('video_placeholder').style.display = 'inline';
            document.getElementById('user_preview').src = "";
            document.getElementById('user_placeholder').style.display = 'inline';
            document.getElementById('user_preview').style.display = 'none';

            document.getElementById('show_logs_button').disabled = true;
            addMessage("System", "Session has been reset.", "system");
        }
    });
}

function changeContext() {
    const contextBackground = document.getElementById('contextBackground');
    const contextWindow = document.getElementById('contextWindow');
    const contextInput = document.getElementById('contextInput');

    fetch('/get_context')
    .then(response => response.json())
    .then(data => {
        if (data.context) {
            contextInput.value = data.context;
        }   else {
            contextInput.value = "";
            contextInput.placeholder = "Enter context here...";
        }
    });
    contextBackground.style.display = 'block';
    contextWindow.style.display = 'block';

    document.getElementById('saveContextBtn').onclick = function() {
        const newContext = contextInput.value;
        const formData = new FormData();
        formData.append('context', newContext);

        fetch('/set_context', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                addMessage("System", "Context has been updated.", "system");
            }
        });

        contextBackground.style.display = 'none';
        contextWindow.style.display = 'none';
    };

    document.getElementById('cancelContextBtn').onclick = function() {
        contextBackground.style.display = 'none';
        contextWindow.style.display = 'none';
    };
}

// ====================
// Database Management
// ====================

function editDatabase() {
    const databaseBackground = document.getElementById('databaseBackground');
    const databaseWindow = document.getElementById('databaseWindow');
    const userWindow = document.getElementById('userWindow');

    let userId = null;
    let userProfile = null;

    databaseBackground.style.display = 'block';
    databaseWindow.style.display = 'block';

    const userImageInput = document.getElementById('userImageInput');

    userImageInput.onchange = function(event) {
        const file = event.target.files[0];
        if (file) {

            const formData = new FormData();
            formData.append('user_image', file);
            fetch('/identify_user', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Optionally handle response, e.g. show a message
                if (data.user) {
                    userId = data.user;
                    userProfile = data.profile;
                }
            });

            const reader = new FileReader();
            reader.onload = function(e) {
                const userPreview = document.getElementById('selected_user_preview');
                userPreview.src = e.target.result;
                userPreview.style.display = 'inline';
            };
            reader.readAsDataURL(file);
            document.getElementById('editUserBtn').style.display = 'inline';
            document.getElementById('userLabel').innerText = 'Change User';

        } else {
            document.getElementById('selected_user_preview').style.display = 'none';
        }
    };

    // USER EDITION
    document.getElementById('editUserBtn').onclick = function() {
            
        databaseWindow.style.display = 'none';
        userWindow.style.display = 'block';
        const editUserImage = document.getElementById('edit_user_image');

        editUserImage.src = document.getElementById('selected_user_preview').src;
        const editUserLabel = document.getElementById('edit_user_label');
        editUserLabel.textContent = userId || "Unknown User";

        populateTableFromProfile(userProfile);

        document.getElementById('saveUserBtn').onclick = function() {
            const formData = new FormData();
            formData.append('user_id', userId);
            
            // Collect data from the table
            const table = document.getElementById("edit_profile");
            const rows = table.querySelectorAll("tbody tr");
            const profileData = [];
            
            rows.forEach(row => {
                const cells = row.querySelectorAll("td");
                const item = {};

                // Determine if the row is a primary or contextual feature
                if (row.classList.contains("primary-feature")) {
                    item.type = "primary";
                } else {
                    item.type = "contextual";
                }
                
                // Map cell values to object properties
                cells.forEach((cell, index) => {
                    if (index === 0) {
                        item.name = cell.textContent;
                    } else if (index === 1) {
                        item.description = cell.textContent;
                    } else if (index === 2) {
                        item.value = cell.textContent.split(',').map(value => value.trim());
                    }
                });
                profileData.push(item);
            });

            formData.append('profile_data', JSON.stringify(profileData));

            fetch('/edit_user', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === "success") {
                    userWindow.style.display = 'none';
                    databaseWindow.style.display = 'block';
                    userProfile = profileData; // Update the userProfile with the new data
                }
            });
        };


        document.getElementById('cancelUserBtn').onclick = function() {
            userWindow.style.display = 'none';
            databaseWindow.style.display = 'block';
        };                                         
    }

    document.getElementById('cancelDatabaseBtn').onclick = function() {

            document.getElementById('userLabel').innerText = 'Select User';
            document.getElementById('editUserBtn').style.display = 'none';
            document.getElementById('selected_user_preview').style.display = 'none';

            databaseBackground.style.display = 'none';
            databaseWindow.style.display = 'none';

    };
}


// Table features work
document.addEventListener("DOMContentLoaded", function () {
    const tbody = document.getElementById("features-table");

    initForm();

    tbody.addEventListener("click", function (event) {
        const clickedRow = event.target.closest("tr");

        if (!clickedRow || clickedRow.classList.contains("expanded-detail")) {
            return; // ignore clicks on detail rows
        }

        clickedRow.classList.toggle("expanded");

        // Close any currently open detail rows
        const openDetailRows = tbody.querySelectorAll(".expanded-detail");
        if (openDetailRows.length) {
            openDetailRows.forEach(openDetailRow => {
                const wrappers = openDetailRow.querySelectorAll(".detail-wrapper");

                wrappers.forEach(wrapper => {
                    wrapper.style.transition = "";
                    wrapper.style.height = wrapper.scrollHeight + "px";
                    wrapper.offsetHeight; // force reflow
                    wrapper.style.transition = "height 0.3s ease, padding 0.3s ease";
                    wrapper.style.height = "0px";
                    wrapper.style.padding = "0 10px";
                });

                wrappers[0].addEventListener("transitionend", function removeRows(e) {
                    if (e.propertyName === "height") {
                        openDetailRows.forEach(row => row.remove());
                        wrappers[0].removeEventListener("transitionend", removeRows);
                    }
                });
            });

            // If clicking the same row that was open, just close and stop here
            const firstOpenRow = openDetailRows[0];
            if (firstOpenRow.previousElementSibling === clickedRow) {
                clickedRow.classList.remove("expanded");
                return;
            }
            firstOpenRow.previousElementSibling.classList.remove("expanded");
        }

        const colCount = clickedRow.children.length;

        // === 1) Create editable detail row ===
        const detailRow = document.createElement("tr");
        detailRow.classList.add("expanded-detail");

        for (let i = 0; i < colCount; i++) {
            const td = document.createElement("td");

            const wrapper = document.createElement("div");

            detailRow.style.marginBottom = "0px";
            wrapper.className = "detail-wrapper";
            wrapper.style.height = "0px";
            wrapper.style.padding = "0px";
            wrapper.style.overflow = "hidden";
            wrapper.style.transition = "height 0.3s ease, padding 0.3s ease";

            const cell_value = clickedRow.children[i].textContent;
            wrapper.innerHTML = `<textarea rows="1" style="width: 100%;">${cell_value}</textarea>`;

            td.appendChild(wrapper);
            detailRow.appendChild(td);

            // Animate open
            requestAnimationFrame(() => {
                wrapper.style.height = wrapper.scrollHeight + "px";
                wrapper.style.padding = "0px";
            });
        }

        clickedRow.insertAdjacentElement("afterend", detailRow);

        // === 2) Create Save button row ===
        const saveRow = document.createElement("tr");
        saveRow.classList.add("expanded-detail");

        const saveTd = document.createElement("td");
        saveTd.setAttribute("colspan", colCount);

        const saveWrapper = document.createElement("div");
        saveTd.style.padding = "0px";
        saveWrapper.className = "detail-wrapper";
        saveWrapper.style.height = "0px";
        saveWrapper.style.padding = "0px";
        saveWrapper.style.overflow = "hidden";
        saveWrapper.style.transition = "height 0.3s ease, padding 0.3s ease";

        if (clickedRow.classList.contains("primary-feature")) {
            saveWrapper.innerHTML = `
                <div class="button-row">
                    <button id="save-feature-btn" class="green_button">Save feature</button>
                </div>
            `;
        } else {
            saveWrapper.innerHTML = `
                <div class="button-row">
                    <button id="save-feature-btn" class="green_button">Save feature</button>
                    <button id="delete-feature-btn" class="red_button">Delete feature</button>
                </div>
            `;
        }
        

        saveTd.appendChild(saveWrapper);
        saveRow.appendChild(saveTd);
        detailRow.insertAdjacentElement("afterend", saveRow);

        // Animate save row open
        requestAnimationFrame(() => {
            saveWrapper.style.height = saveWrapper.scrollHeight + "px";
            saveWrapper.style.padding = "0px";
        });

        saveWrapper.querySelector("#save-feature-btn").onclick = function () {
            const detailTextareas = detailRow.querySelectorAll("textarea");
            for (let i = 0; i < detailTextareas.length; i++) {
                clickedRow.children[i].textContent = detailTextareas[i].value;
            }

            // Close the detail row after saving
            detailRow.querySelectorAll(".detail-wrapper").forEach(wrapper => {
                wrapper.style.transition = "";
                wrapper.style.height = wrapper.scrollHeight + "px";
                wrapper.offsetHeight; // force reflow
                wrapper.style.transition = "height 0.3s ease, padding 0.3s ease";
                wrapper.style.height = "0px";
                wrapper.style.padding = "0px";
            });

            detailRow.remove();
            saveRow.remove();
            clickedRow.classList.remove("expanded");

        };

        saveWrapper.querySelector("#delete-feature-btn").onclick = function () {
            // Remove the clicked row and its detail rows
            clickedRow.remove();
            detailRow.remove();
            saveRow.remove();

            // If there are no more rows, reset the table
            if (tbody.children.length === 0) {
                const emptyRow = document.createElement("tr");
                emptyRow.innerHTML = "<td colspan='4'>No features available</td>";
                tbody.appendChild(emptyRow);
            }
        };

        
    });
});

document.getElementById("addFeatureBtn").addEventListener("click", function () {
    const featureWindow = document.getElementById("featureWindow");
    const userWindow = document.getElementById("userWindow");
    featureWindow.style.display = "block";
    userWindow.style.display = "none";

    // Reset the form and pills
    const form = document.getElementById("add-feature-form");
    form.reset();
    document.getElementById("values-container").innerHTML = '<input type="text" id="feature-values-input" placeholder="Type value and press Enter">';
    setupPillInput("values-container", "feature-values-input", "feature-values", ["Enter"]);

    // Update description for the currently selected name
    if (updateDescriptionFunc) updateDescriptionFunc();
});

// Only attach the submit listener once
document.getElementById("add-feature-form").addEventListener("submit", function (e) {
    e.preventDefault();
    // check if a row exists with the same name
    const existingRow = Array.from(document.getElementById("features-table").children).find(row => row.children[0].textContent === document.getElementById("feature-name").value);
    if (existingRow) {
        // Update the value of the existing row
        existingRow.children[2].textContent = existingRow.children[2].textContent + " ; " + document.getElementById("feature-values").value;
    } else {
        const newRow = document.createElement("tr");
        newRow.innerHTML = `
            <td>${document.getElementById("feature-name").value}</td>
            <td>${document.getElementById("feature-description").value}</td>
            <td>${document.getElementById("feature-values").value}</td>
        `;
        document.getElementById("features-table").appendChild(newRow);
    }
    document.getElementById("featureWindow").style.display = "none";
    document.getElementById("userWindow").style.display = "block";
});

document.getElementById("cancel-feature-btn").addEventListener("click", function () {
    featureWindow.style.display = "none";
    userWindow.style.display = "block";
    document.getElementById("add-feature-form").reset();
    // reset pill containers
    document.getElementById("values-container").innerHTML = '<input type="text" id="feature-values-input" placeholder="Type value and press Enter">';
});

let updateDescriptionFunc = null; // global reference

async function initForm() {
    const res = await fetch("/config");
    const config = await res.json();

    const nameContainer = document.getElementById("name-container");
    const descriptionInput = document.getElementById("feature-description");

    if (config.memory.closed_vocabulary === false) {
        // Free text input
        nameContainer.innerHTML = `
            <input type="text" id="feature-name" name="feature-name"
                pattern="^[A-Za-z0-9_\\-]+$"
                title="Only letters, numbers, underscores, and hyphens allowed"
                placeholder="Type feature name here..." required>`;
        descriptionInput.removeAttribute("readonly");
        descriptionInput.placeholder = "Type description here...";
        updateDescriptionFunc = null; // nothing to update dynamically
    } else if (config.memory.closed_vocabulary === true) {
        // Dropdown select
        let options = Object.entries(config.memory.vocabulary)
            .map(([name]) => `<option value="${name}">${name}</option>`)
            .join("");
        nameContainer.innerHTML = `<select id="feature-name" name="feature-name" required>${options}</select>`;

        const nameSelect = document.getElementById("feature-name");

        function updateDescription() {
            descriptionInput.value = config.memory.vocabulary[nameSelect.value] || "";
            descriptionInput.setAttribute("readonly", true);
        }

        // Trigger update on change
        nameSelect.addEventListener("change", updateDescription);

        // Expose globally so we can call it on form open
        updateDescriptionFunc = updateDescription;

        // initial update
        updateDescription();
    }
}

function setupPillInput(containerId, inputId, hiddenId, triggerKeys) {
    const container = document.getElementById(containerId);
    const input = document.getElementById(inputId);
    const hidden = document.getElementById(hiddenId);
    let items = [];

    function createPill(text) {
        const pill = document.createElement("div");
        pill.className = "pill";
        pill.textContent = text;

        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = "×";
        btn.onclick = () => {
            items = items.filter(i => i !== text);
            pill.remove();
            hidden.value = items.join(",");
        };

        pill.appendChild(btn);
        return pill;
    }

    input.addEventListener("keydown", (e) => {
        if (triggerKeys.includes(e.key) && input.value.trim() !== "") {
            e.preventDefault();
            const value = input.value.trim();
            if (!items.includes(value)) {
                items.push(value);
                container.insertBefore(createPill(value), input);
                hidden.value = items.join(",");
            }
            input.value = "";
        }
    });

    container.addEventListener("click", () => input.focus());
}      


// ====================
// Clean database
// ====================

function verifCleanDatabase() {
    const verifBackground = document.getElementById('verifBackground');
    const verifWindow = document.getElementById('verifWindow');

    verifBackground.style.display = 'block';
    verifWindow.style.display = 'block';

    document.getElementById('confirmResetBtn').onclick = function() {
        resetDatabase();
        verifBackground.style.display = 'none';
        verifWindow.style.display = 'none';
    };

    document.getElementById('cancelResetBtn').onclick = function() {
        verifBackground.style.display = 'none';
        verifWindow.style.display = 'none';
    };
}

function resetDatabase() {
    fetch('/reset_database', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            resetSession();
        }
    });
}
