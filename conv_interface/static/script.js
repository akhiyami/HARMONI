//dynamic style
const user_field= document.getElementById("user");
const profile = document.getElementById("profile");
profile.style.height = `${user_field.offsetHeight}px`;

//constants
const chatbox = document.getElementById("chatbox");
const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const live_profile = document.getElementById("live_profile");

//global variables
let userId = ""; // Default user ID

// Empty the chat form input
questionInput.value = ""

// Event listener for form submission
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    const user = userId; 

    questionInput.value = "";

    if (!user) {
        addMessage("Erreur", "Veuillez sélectionner un utilisateur actif.", "error");
        questionInput.value = ""
        return;
    }

    if (!question) return;

    addMessage(user, question, "user");

    const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({question, user})
    });

    const data = await response.json();
    if (data.answer) {
        addMessage("Assistant", data.answer, "bot");
    } else {
        addMessage("Erreur", data.error || "Erreur inconnue", "bot");
    }

    if (!data.profile) {
        live_profile.innerHTML = "";
    } else {
        // Separate primary and contextual items
        const primary = data.profile.filter(item => item.type === "primary");
        const contextual = data.profile.filter(item => item.type === "contextual");

        let html = "";
        if (primary.length > 0) {
            html += primary.map(
                item => `<b>${item.name}:</b> <span style="color:gray"><em>(${item.description})</em></span> </br>${item.value} `
            ).join('<br>');
        }
        if (contextual.length > 0) {
            if (primary.length > 0) html += "<br>----------<br>";
            html += contextual.map(
                item => `<b>${item.name}:</b> <span style="color:gray"><em>(${item.description})</em></span> </br>${item.value} `
            ).join('<br>');
        }
        live_profile.innerHTML = html;
    }
    
});

// Function to add a message to the chatbox
function addMessage(sender, text, role) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = `${sender}: ${text}`;
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

// Function to handle image selection and update user profile
function handleImageSelect(event) {
    const file = event.target.files[0];
    const preview = document.getElementById('preview');
    const userPlaceholder = document.getElementById('user_placeholder');
    const editButton = document.getElementById('editDatabaseBtn');
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            
            if (userPlaceholder.style.display !== 'none') {
                preview.style.display = 'inline';
                userPlaceholder.style.display = 'none';
                editButton.disabled = false; // Enable the edit button
            }

            const formData = new FormData();
            formData.append('image', file);

            fetch('/set_user', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                addMessage("Change user", data.user_id, "system");
                userId = data.user_id; 

                if (!data.profile) {
                    live_profile.innerHTML = "";
                } else {
                    const primary = data.profile.filter(item => item.type === "primary");
                    const contextual = data.profile.filter(item => item.type === "contextual");

                    let html = "";
                    if (primary.length > 0) {
                        html += primary.map(
                            item => `<b>${item.name}:</b> <span style="color:gray"><em>(${item.description})</em></span> </br>${item.value} `
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
                
            })
            .catch(error => {
                console.error('Error setting user:', error);
                addMessage("Erreur", "Erreur lors de la mise à jour de l'utilisateur actif.", "error");
            });
        };
        reader.readAsDataURL(file);
    } else {
        preview.src = '';
        preview.style.display = 'none';
    }
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

function resetSession() {
    fetch('/reset_session', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            chatbox.innerHTML = "";

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
// DATABASE EDITION
// ====================

async function editDatabase() {
    const databaseBackground = document.getElementById('databaseBackground');
    const userWindow = document.getElementById('userWindow');

    let userProfile = null; 

    databaseBackground.style.display = 'block';
    userWindow.style.display = 'block';

    async function loadProfile() {
        const formData = new FormData();
        formData.append('user_id', userId);

        const response = await fetch('/get_profile', { method: 'POST', body: formData });
        const data = await response.json();
        return data.profile;
    }

    userProfile = await loadProfile();

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
                databaseBackground.style.display = 'none';
                userProfile = profileData; // Update the userProfile with the new data

                const primary = userProfile.filter(item => item.type === "primary");
                const contextual = userProfile.filter(item => item.type === "contextual");

                let html = "";
                if (primary.length > 0) {
                    html += primary.map(
                        item => `<b>${item.name}:</b> <span style="color:gray"><em>(${item.description})</em></span> </br>${item.value} `
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
        });
    };

    document.getElementById('cancelUserBtn').onclick = function() {
        userWindow.style.display = 'none';
        databaseBackground.style.display = 'none';
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
