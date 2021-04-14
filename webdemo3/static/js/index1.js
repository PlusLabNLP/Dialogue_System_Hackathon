var $messages = $('.messages-content');
var serverResponse = "wala";


var suggession;
//speech reco
try {
  var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  var recognition = new SpeechRecognition();
}
catch(e) {
  console.error(e);
  $('.no-browser-support').show();
}

$('#start-record-btn').on('click', function(e) {
  recognition.start();
});

recognition.onresult = (event) => {
  const speechToText = event.results[0][0].transcript;
 document.getElementById("MSG").value= speechToText;
  //console.log(speechToText)
  insertMessage()
}

function convline_words_edit(idx) {
  let status = document.getElementById(`conv-words-${idx}`).style.pointerEvents;
  let value = document.getElementById(`conv-words-${idx}`).value;
  if (status == "none") {
    document.getElementById(`conv-words-${idx}`).style.pointerEvents = "auto";
    document.getElementById(`conv-words-${idx}`).style.backgroundColor = "ghostwhite";
    document.getElementById(`conv-words-${idx}`).style.fontStyle = "normal";
    document.getElementById(`convline_word_edit_icon-${idx}`).style.color = "lawngreen";
  } else {
    document.getElementById(`conv-words-${idx}`).style.pointerEvents = "none";
    document.getElementById(`conv-words-${idx}`).style.backgroundColor = "";
    document.getElementById(`conv-words-${idx}`).style.fontStyle = "italic";
    document.getElementById(`convline_word_edit_icon-${idx}`).style.color = "green";
    CheckRemove(idx, value);
  }
}
function convline_words_accept(idx) {
  document.getElementById(`conv-words-${idx}`).style.pointerEvents = "none";
  document.getElementById(`conv-words-${idx}`).style.backgroundColor = "";
  document.getElementById(`conv-words-${idx}`).style.fontStyle = "italic";
  let value = document.getElementById(`conv-words-${idx}`).value;
  CheckRemove(idx, value);
  document.getElementById(`convline_word_edit_icon-${idx}`).style.color = "#666565";
  document.getElementById(`convline_word_edit-${idx}`).style.pointerEvents = "none";
  document.getElementById(`convline_word_accept_icon-${idx}`).style.color = "#f0ff13";
  document.getElementById(`convline_word_accept-${idx}`).style.pointerEvents = "none";
}
function CheckRemove(idx, value) {
  if (value == "") {
    document.getElementById(idx).remove()
  }
}

function convline_template(idx) {
  let template = [`
    <div id="${idx}">
        <input class="convline-word" id="conv-words-${idx}" type="text" value="none" style="pointer-events: auto;font-style: normal;background-color: ghostwhite;">
        <button class="convline-edit-btn" type="button" id="convline_word_edit-${idx}"><i class="fas fa-edit"
        aria-hidden="true" id="convline_word_edit_icon-${idx}" title="Edit"
        style="color: lawngreen;"></i></button>
        <button class="convline-confirm-btn" type="button" id="convline_word_accept-${idx}"><i
        class="fas fa-check-double" aria-hidden="true" title="Accept" id="convline_word_accept_icon-${idx}"
        style="color:aliceblue;"></i></button>
    </div>`,
  `
      
        $("#convline_word_edit-${idx}").click(function () {
            let idx = $(this).attr('id').split('-')[1];
            //console.log(idx.split('-')[1]);
            let value = document.getElementById("conv-words-${idx}").value;
            convline_words_edit(idx, value);
        })
        $("#convline_word_accept-${idx}").click(function () {
            let idx = $(this).attr('id').split('-')[1];
            if (document.getElementById("main-message-box").getAttribute("firstacceptclick") == "true") {
            if (confirm("Warning: once click Accept button, you cannot change the ConvLine word anymore! (This message won't show later. )")) {
                document.getElementById("main-message-box").setAttribute("firstacceptclick", "false");
                convline_words_accept(idx);
            }
            }else{
            convline_words_accept(idx);
            }
        })
      
    `]
  return template;
}

function addConvlineWords(itm_idx) {
  let el1 = document.getElementById(itm_idx).parentNode;
  let idx = el1.previousElementSibling.getAttribute('id');
  number = parseInt(idx.split('_')[1]);
  number += 1;
  number = number.toString();
  let idx1 = [idx.split('_')[0], number].join('_')
  let template = convline_template(idx1);
  el1.insertAdjacentHTML('beforebegin', template[0]);
  let childNode = document.createElement('script')
  childNode.innerHTML = template[1]
  document.getElementById(idx1).appendChild(childNode)
}

function ConfirmAll(idx) {
  if (document.getElementById(idx).getAttribute("confirmall") == "false" && document.getElementById(idx).getAttribute("finish_generation") == "true") {
    let buttons = document.getElementById(idx).parentElement.parentElement.getElementsByClassName('convline-confirm-btn');
    for (i = 0; i < buttons.length; i++) {
      buttons[i].click();
    }
    let num = idx.split('-')[1];
    let plusIdx = [num, 'plus'].join('_');
    //document.getElementById(plusIdx).style.visibility = "hidden";
    document.getElementById(plusIdx).remove();
    let icon = ['confirm_all_btn_icon', num].join('-');
    document.getElementById(icon).style.color = "yellow";
    document.getElementById(idx).style.pointerEvents = "none";
    document.getElementById(idx).setAttribute("confirmall", "true");
    document.getElementById(idx).setAttribute("finish_generation", "false");
  } else {
    console.log(1);
  }

}

function listendom(no){
  console.log(no)
  //console.log(document.getElementById(no))
document.getElementById("MSG").value= no.innerHTML;
  insertMessage();
}

$(window).load(function() {
  $messages.mCustomScrollbar();
  setTimeout(function() {
    serverMessage("hello i am customer support bot type hi and i will show you quick buttions");
  }, 100);

});

function updateScrollbar() {
  $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
    scrollInertia: 10,
    timeout: 0
  });
}



function insertMessage() {
  msg = $('.message-input').val();
  if ($.trim(msg) == '') {
    return false;
  }
  $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
  // fetchmsg() 
  
  $('.message-input').val(null);
  updateScrollbar();

}

document.getElementById("mymsg").onsubmit = (e)=>{
  e.preventDefault() 
  insertMessage();
  serverMessage("hello");
  //speechSynthesis.speak( new SpeechSynthesisUtterance("hello"))
}

function serverMessage(response2) {


  if ($('.message-input').val() != '') {
    return false;
  }
  $('<div class="message loading new"><figure class="avatar"><img src="css/bot.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
  updateScrollbar();
  debugger;

  setTimeout(function() {
    $('.message.loading').remove();
    $('<div class="message new"><figure class="avatar"><img src="css/bot.png" /></figure>' + response2 + '</div>').appendTo($('.mCSB_container')).addClass('new');
    updateScrollbar();
  }, 100 + (Math.random() * 20) * 100);

}


function fetchmsg(){

     var url = 'http://localhost:5000/send-msg';
      
      const data = new URLSearchParams();
      for (const pair of new FormData(document.getElementById("mymsg"))) {
          data.append(pair[0], pair[1]);
          console.log(pair)
      }
    
      console.log("abc",data)
        fetch(url, {
          method: 'POST',
          body:data
        }).then(res => res.json())
         .then(response => {
          console.log(response);
        //  serverMessage(response.Reply);
          //speechSynthesis.speak( new SpeechSynthesisUtterance(response.Reply))
        
          
         })
          .catch(error => console.error('Error h:', error));

}


