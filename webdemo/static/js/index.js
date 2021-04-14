var $messages = $('.messages-content');
var serverResponse = "wala";
var message_list = {'chatbot':{},'user':{}};
var turn_number = 0;

var suggession;
//speech reco

var convline_status = "on"

function convline_onoff(action) {
  if (action=='turn-on') {
    //$('.convline').last().removeClass('close-convline')
    //$('.final-confirm-btn').last().removeClass('close-final-btn')
    convline_status = "on"
    //let num = document.getElementsByClassName('message new').length;
    //document.getElementsByClassName('message new')[num-1].remove();
    //turnRequest();
    //$("#MSG").focus().select()
  } else {
    //$('.convline').last().addClass('close-convline')
    //$('.final-confirm-btn').last().addClass('close-final-btn')
    convline_status = "off"
    //let num = document.getElementsByClassName('message new').length;
    //document.getElementsByClassName('message new')[num-1].remove();
    //turnRequest();
    ///$("#MSG").focus().select()
  }
  if (document.getElementsByClassName('final-confirm-btn')[document.getElementsByClassName('final-confirm-btn').length-1].attributes['confirmed'].value=='false'){
    if (confirm("Regenerate current response again?")) {
      let num = document.getElementsByClassName('message new').length;
      document.getElementsByClassName('message new')[num-1].remove();
      turnRequest();
      $("#MSG").focus().select()
    } else {
      //document.getElementById("main-message-box").setAttribute("firstacceptclick", "false");
      if (action=='turn-on') {
        $('.convline').last().removeClass('close-convline');
        $('.final-confirm-btn').last().removeClass('close-final-btn');
        $('.convline-confirm-all-btn').last().removeClass('close-final-btn')
      } else {
        $('.convline').last().addClass('close-convline')
        $('.final-confirm-btn').last().addClass('close-final-btn')
        $('.convline-confirm-all-btn').last().addClass('close-final-btn')
      }
    }
  }
}

function addArrow() {
  let node1 = document.createElement('div');
  node1.innerHTML = '<div class="confirm-pic" style="width: 0;"><img src="static/img/red-arrow-confirm.png" style="height: 2rem;width: 5rem;"></div>';
  $('.responseline').last()[0].appendChild(node1)
  let node2 = document.createElement('div');
  node2.innerHTML = `<img src='static/img/red-arrow-send.png' style="position: absolute;top:-6rem;left: 58rem;height: 5rem;width: 2rem;" class="confirm-pic">`;
  $("#mymsg")[0].appendChild(node2)
}

function removeArrow() {
  $(".confirm-pic").remove()
}

function convline_words_edit(idx) {
  let status = document.getElementById(`conv-words-${idx}`).style.pointerEvents;
  let value = document.getElementById(`conv-words-${idx}`).value;
  if (status == "none") {
    document.getElementById(`conv-words-${idx}`).style.pointerEvents = "auto";
    document.getElementById(`conv-words-${idx}`).style.backgroundColor = "ghostwhite";
    document.getElementById(`conv-words-${idx}`).style.fontStyle = "normal";
    document.getElementById(`convline_word_edit_icon-${idx}`).style.color = "lawngreen";
    const text = document.getElementById(`conv-words-${idx}`)
    text.focus()
    text.select()
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
  value = value.trim();
  if (value == "" || value=="<none>") {
    document.getElementById(idx).remove()
  }
}

function convline_template(idx, word=null) {
  let tokens;
  if (word == null) {
    tokens = '<none>';
  } else {
    tokens = word.trim();
  }
  let template = [`
    <div id="${idx}" class="convline-sample">
        <input class="convline-word" id="conv-words-${idx}" type="text" value="${tokens}" style="pointer-events: auto;font-style: normal;background-color: ghostwhite;">
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
            //if (document.getElementById("main-message-box").getAttribute("firstacceptclick") == "true") {
              //if (confirm("Warning: once click Accept button, you cannot change the ConvLine word anymore! (This message won't show later. )")) {
                  //document.getElementById("main-message-box").setAttribute("firstacceptclick", "false");
            convline_words_accept(idx);
              //} else {
                //document.getElementById("main-message-box").setAttribute("firstacceptclick", "false");
              //}
            //}else{
            //convline_words_accept(idx);
            //}
        })
      
    `]
  return template;
}

function addConvlineWords(itm_idx, word=null, flag=null) {
  let el1 = document.getElementById(itm_idx).parentNode;
  let el2 = el1.parentNode;
  if (el2.querySelector('.convline-word')==null) {
    let turn_num = itm_idx.split('-')[1];
    let template = convline_template(`${turn_num}_1`, word);
    el1.insertAdjacentHTML('beforebegin', template[0]);
    let childNode = document.createElement('script')
    childNode.innerHTML = template[1]
    document.getElementById(`${turn_num}_1`).appendChild(childNode)
    if (flag!=null && word!='' && word!=null) {
      convline_words_edit(`${turn_num}_1`);
    } else {
      let text = document.getElementById(`conv-words-${turn_num}_1`)
      text.focus();
      text.select();
    }
  } else {
    let idx = el1.previousElementSibling.getAttribute('id');
    number = parseInt(idx.split('_')[1]);
    number += 1;
    number = number.toString();
    let idx1 = [idx.split('_')[0], number].join('_')
    let template = convline_template(idx1, word);
    el1.insertAdjacentHTML('beforebegin', template[0]);
    let childNode = document.createElement('script')
    childNode.innerHTML = template[1]
    document.getElementById(idx1).appendChild(childNode)
    if (flag!=null && word!='' && word!=null) {
      convline_words_edit(idx1);
    } else {
      let text = document.getElementById(`conv-words-${idx1}`)
      text.focus();
      text.select();
    }
  }
}

function ConfirmAll(idx) {
  let num = idx.split('-')[1];
  disable_convline('final_confirm', num)
  document.getElementById(idx).setAttribute("confirmed", "true");
  document.getElementById(`final_confirm_btn_icon-${num}`).style.color = 'cornflowerblue';
  document.getElementById(idx).style.pointerEvents = "none";
}

function disable_convline(state, index) {
  let final_confirm_id = ["final_confirm_btn", index].join('-')
  let regenerate_btn_id = ["confirm_all_btn", index].join('-')
  keywords_regen_btn_id = ["keywords_regenerate_btn", index].join('-')
  console.assert(document.getElementById(regenerate_btn_id).getAttribute("confirmall") == "false")
  console.assert(document.getElementById(regenerate_btn_id).getAttribute("finish_generation") == "true")
  console.assert(document.getElementById(final_confirm_id).getAttribute("confirmed")=="false")

  let buttons = document.getElementById(regenerate_btn_id).parentElement.parentElement.parentElement.querySelector(`#convline_${index}`).getElementsByClassName('convline-confirm-btn');
  if (buttons!=null) {
    for (i = 0; i < buttons.length; i++) {
      buttons[i].click();
    }
  }
  
  let num = regenerate_btn_id.split('-')[1];
  let plusIdx = [num, 'plus'].join('_');
  let icon = ['confirm_all_btn_icon', num].join('-');
  let icon1 = ["keywords_regenerate_btn_icon", num].join('-');
  //document.getElementById(plusIdx).style.visibility = "hidden";
  if (state=="final_confirm"){
    document.getElementById(plusIdx).remove();
    document.getElementById(icon).style.color = "gray";
    document.getElementById(regenerate_btn_id).style.pointerEvents = "none";
    document.getElementById(regenerate_btn_id).setAttribute("confirmall", "true");
    document.getElementById(regenerate_btn_id).setAttribute("finish_generation", "false");
    document.getElementById(icon1).style.color = "gray";
    document.getElementById(keywords_regen_btn_id).style.pointerEvents = "none";
  } else if(state=="regeneration") {
    document.getElementById(plusIdx).hidden=true;
    document.getElementById(final_confirm_id).hidden=true;
    //document.getElementById(icon).style.color = "#fefea7";
    //document.getElementById(regenerate_btn_id).style.pointerEvents = "none";
    document.getElementById(regenerate_btn_id).hidden=true;
    document.getElementById(regenerate_btn_id).setAttribute("confirmall", "true");
    document.getElementById(icon1).style.color = "#fefea7";
    document.getElementById(keywords_regen_btn_id).style.pointerEvents = "none";

  } else if(state=='keywords-regeneration') {
    document.getElementById(icon).style.color = "#fefea7";
    document.getElementById(regenerate_btn_id).style.pointerEvents = "none";
    document.getElementById(plusIdx).hidden=true;
    document.getElementById(keywords_regen_btn_id).hidden=true;
    document.getElementById(`final_confirm_btn_icon-${num}`).style.color = "#fefea7";
    document.getElementById(final_confirm_id).style.pointerEvents = "none";

  }
}

function regenerate(idx) {
  let num = idx.split('-')[1];
  disable_convline('regeneration', num)
  let nodes = document.getElementById("mCSB_1_container").lastChild.querySelectorAll(".convline-word");
  let convline_words = [];
  if (nodes!=null) {
    for (i=0; i<nodes.length;i++) {
      convline_words.push(nodes[i].value);
    }
  }
  let msg_nodes = document.getElementsByClassName("message")
  let history = []
  for (i=0;i<msg_nodes.length;i++){
    history.push(msg_nodes[i].querySelector(".message-text").innerHTML)
  }
  sendData = {
    "history": history,
    "keywords": convline_words,
    "temperature": document.getElementById('temperature').value,
    "top-k": document.getElementById('top-k').value,
    "top-p": document.getElementById('top-p').value
  }
  
  sendData = JSON.stringify(sendData);
  // generate animate for message waiting
  let text_node = document.getElementById(`confirm_all_btn-${num}`).parentNode.parentNode.parentNode.querySelector(".message-text");
  text_node.classList.add("message1");
  text_node.classList.add("loading");
  text_node.classList.add("new");
  text_node.classList.remove("message-text");
  text_node.innerHTML=`<span></span>`;
  $.ajax({
    url:'/single-rerun/',
    method: 'POST',
    dataType: 'json',
    contentType: 'application/json;charset=utf-8',
    data: sendData,
    async: true,
    timeout: 0,
    crossDomain: true,
    success: function (data) {
      console.log('receive api request')
      console.log(data)
    },
    error: function(jqxhr, status, exception) {
      console.log('Exception:', JSON.stringify(exception, null, 2));
      console.log('fail to get request response')
      $(".notify1").remove("active")
      $("#notifyType1").remove("sending")
      $(".notify1").addClass("error")
      $(".notify1").addClass("active")
      $(".notify1").attr("color","red")
      $("#MSG").removeAttr('disabled')
      $("#msg-btn").css("background-color", "#248A52")
      $("#msg-btn").removeClass("not-changable")
      $("#MSG").attr("placeholder",'type message...')
      $(".parameter-group").removeClass("not-allow")
      $(".form-control-range").removeClass("not-changable")
      $(".submission-group").removeClass("not-allow")
      $(".submit-button").removeClass("not-changable")
      $("#convline-onoff").removeClass("not-changable")
      setTimeout(function(){
        $(".notify1").removeClass("active");
        $("#notifyType").removeClass("error");
        $(".notify1").setAttribute("color","white")
      },10000);
    },
    beforeSend: function() {
      $(".notify1").remove("active")
      $("#notifyType1").addClass("sending")
      $(".notify1").addClass("active")
      $("#MSG").attr('disabled','disabled');
      $("#msg-btn").css("background-color", "gray")
      $("#msg-btn").addClass("not-changable")
      $("#MSG").attr("placeholder",'🚫')
      //$('<div class="message loading new"><figure class="avatar"><img src="static/css/pluslab-icon.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
      $(".parameter-group").addClass("not-allow")
      $(".form-control-range").addClass("not-changable")
      $(".submission-group").addClass("not-allow")
      $(".submit-button").addClass("not-changable")
      $("#convline-onoff").addClass("not-changable")
      updateScrollbar();
    }, 
    complete: function(data) {
      $(".notify1").remove("active")
      $("#notifyType1").remove("sending")
      $(".notify1").addClass("getMsg")
      $(".notify1").addClass("active")
      $("#MSG").removeAttr('disabled')
      $("#msg-btn").css("background-color", "#248A52")
      $("#msg-btn").removeClass("not-changable")
      $("#MSG").attr("placeholder",'type message...')
      $(".parameter-group").removeClass("not-allow")
      $(".form-control-range").removeClass("not-changable") 
      $(".submission-group").removeClass("not-allow")
      $(".submit-button").removeClass("not-changable")
      $("#convline-onoff").removeClass("not-changable")
      setTimeout(function(){
        $(".notify1").removeClass("active");
        $("#notifyType").removeClass("sending");
      },1000);
      //$('.message.loading').remove();
      //if (typeof data === 'string'){
      //  data = JSON.parse(data);
      //  console.log(22)
      //}
      let responses = data.responseJSON["responses"];
      let convlines = data.responseJSON["keywords"].split("#");
      for (i=0;i<convlines.length;i++) {
        convlines[i] = convlines[i].trim();
      }
      console.log(data.responseJSON)
      console.log(responses)
      console.log(convlines)
      document.getElementById(`${num}_plus`).hidden=false;
      document.getElementById(`final_confirm_btn-${num}`).hidden=false;
      document.getElementById(`keywords_regenerate_btn_icon-${num}`).style.color = "aliceblue";
      document.getElementById(`keywords_regenerate_btn_icon-${num}`).style.pointerEvents = "auto";
      document.getElementById(`confirm_all_btn-${num}`).setAttribute("confirmall", "false");
      document.getElementById(`confirm_all_btn-${num}`).hidden=false;
      let past_convline = document.getElementById(`confirm_all_btn-${num}`).parentNode.parentNode.parentNode.querySelectorAll('.convline-sample');
      if (past_convline.length>0) {
        for (i=0;i<past_convline.length;i++) {
          past_convline[i].remove();
        }
      }
      for (i=0;i<convlines.length;i++) {
        addConvlineWords(`convline_plus-${num}`, convlines[i], true);
      }
      text_node.classList.add("message-text");
      text_node.classList.remove("message1");
      text_node.classList.remove("loading");
      text_node.classList.remove("new");
      text_node.innerHTML=responses[0];
      let idx = parseInt(num)
      message_list["chatbot"][idx].push([convlines, responses[0], [document.getElementById('temperature').value, document.getElementById('top-k').value,  document.getElementById('top-p').value]])
    }
  });
}

$(window).load(function () {
  turn_number += 1;
  $messages.mCustomScrollbar();
  setTimeout(function () {
    serverMessage("Hi! I'm DiSCoL Bot. ", [], 1);
  }, 50);

});

function updateScrollbar() {
  console.log("update scrollbar begin")
  $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
    scrollInertia: 10,
    timeout: 0
  });
  console.log("update scrollbar end")
}



function insertMessage() {
  msg = $('.message-input').val();
  if ($.trim(msg) === '') {
    return false;
  }
  turn_number += 1;
  let turn_num = document.getElementsByClassName("message").length;
  turn_num += 1;
  $(`<div class="message message-personal" id="message-${turn_num}"><div class="message-text">` + msg + '</div></div>').appendTo($('.mCSB_container')).addClass('new');
  
  $('.message-input').val(null);
  updateScrollbar();
  message_list["user"][turn_num] = [msg];
}

function turnRequest() {
  console.log("turnRequest");
  let nodes = document.getElementsByClassName("message")
  let history = []
  for (i=0;i<nodes.length;i++){
    history.push(nodes[i].querySelector(".message-text").innerHTML)
  }
  sendData = {
    "history": history,
    "temperature": document.getElementById('temperature').value,
    "top-k": document.getElementById('top-k').value,
    "top-p": document.getElementById('top-p').value,
    "convline-onoff": document.getElementById('convline-onoff').attributes['convline'].nodeValue
  }
  sendData = JSON.stringify(sendData);
  $.ajax({
    url:'/generate/',
    method: 'POST',
    dataType: 'json',
    contentType: 'application/json;charset=utf-8',
    data: sendData,
    async: true,
    timeout: 0,
    crossDomain: true,
    success: function (data) {
      console.log('receive api request')
      console.log(data)
    },
    error: function(jqxhr, status, exception) {
      console.log('Exception:', JSON.stringify(exception, null, 2));
      console.log('fail to get request response')
      $(".notify1").remove("active")
      $("#notifyType1").remove("sending")
      $(".notify1").addClass("error")
      $(".notify1").addClass("active")
      $(".notify1").attr("color","red")
      $("#MSG").removeAttr('disabled')
      $("#msg-btn").css("background-color", "#248A52")
      $("#msg-btn").removeClass("not-changable")
      $("#MSG").attr("placeholder",'type message...')
      $(".parameter-group").removeClass("not-allow")
      $(".form-control-range").removeClass("not-changable")
      $(".submission-group").removeClass("not-allow")
      $(".submit-button").removeClass("not-changable")
      $("#convline-onoff").removeClass("not-changable")
      $('.message.loading').remove();
      setTimeout(function(){
        $(".notify1").removeClass("active");
        $("#notifyType").removeClass("error");
        $(".notify1").setAttribute("color","white")
      },10000);
    },
    beforeSend: function() {
      $(".notify1").remove("active")
      $("#notifyType1").addClass("sending")
      $(".notify1").addClass("active")
      $("#MSG").attr('disabled','disabled');
      $("#msg-btn").css("background-color", "gray")
      $("#msg-btn").addClass("not-changable")
      $("#MSG").attr("placeholder",'🚫')
      $('<div class="message loading new"><figure class="avatar"><img src="static/css/pluslab-icon.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
      $(".parameter-group").addClass("not-allow")
      $(".form-control-range").addClass("not-changable")
      $(".submission-group").addClass("not-allow")
      $(".submit-button").addClass("not-changable")
      $("#convline-onoff").addClass("not-changable")
      updateScrollbar();
    }, 
    complete: function(data) {
      $(".notify1").remove("active")
      $("#notifyType1").remove("sending")
      $(".notify1").addClass("getMsg")
      $(".notify1").addClass("active")
      $("#MSG").removeAttr('disabled')
      $("#msg-btn").css("background-color", "#248A52")
      $("#msg-btn").removeClass("not-changable")
      $("#MSG").attr("placeholder",'type message...')
      $(".parameter-group").removeClass("not-allow")
      $(".form-control-range").removeClass("not-changable") 
      $(".submission-group").removeClass("not-allow")
      $(".submit-button").removeClass("not-changable")
      $("#convline-onoff").removeClass("not-changable")
      setTimeout(function(){
        $(".notify1").removeClass("active");
        $("#notifyType").removeClass("sending");
      },1000);
      $('.message.loading').remove();
      //if (typeof data === 'string'){
      //  data = JSON.parse(data);
      //  console.log(22)
      //}
      let responses = data.responseJSON["responses"];
      let convlines = data.responseJSON["keywords"].split("#");
      for (i=0;i<convlines.length;i++) {
        convlines[i] = convlines[i].trim();
      }
      console.log(data.responseJSON)
      console.log(responses)
      console.log(convlines)
      serverMessage(responses, convlines, history.length+1);
      updateScrollbar();
      $("#MSG").focus().select()
    }
  }) 

}

document.getElementById("mymsg").onsubmit = (e) => {
  e.preventDefault()
  if (document.getElementById("MSG").value.trim()=="") {
    console.log("prompt alert");
    $(".notify").addClass("active");
    $("#notifyType").addClass("failure");
    
    setTimeout(function(){
      $(".notify").removeClass("active");
      $("#notifyType").removeClass("failure");
    },2000);
    return 0;
  }
  let confirm_nodes = document.getElementsByClassName('final-confirm-btn');
  let status = confirm_nodes.item(confirm_nodes.length-1).getAttribute('confirmed');
  if (status=="false") {
    confirm_nodes.item(confirm_nodes.length-1).click();
  }
  //document.getElementById("mymsg").style.pointerEvents = "none";
  //document.getElementById("mymsg").disabled="true";
  insertMessage();
  turnRequest();
  $("#MSG").focus().select()
  
}

function serverMessage(response, convlines, num) {
  message_list["chatbot"][num] = [];
  message_list["chatbot"][num].push([convlines, response, [document.getElementById('temperature').value, document.getElementById('top-k').value,  document.getElementById('top-p').value]])
  /*if ($('.message-input').val() != '') {
    return false;
  }*/
  let template_frame = 
  `
    <div class="message" id="message-${num}">
      <figure class="avatar"><img src="static/css/pluslab-icon.png"></figure>
      <form class="convline" method="POST" id="convline_${num}" action="/generate-convline/">
      </form>
      <div class="responseline" id="${num}_responseline"><div class="message-text" style="margin-right:auto">${response} </div> <!--button class="final-confirm-btn" type="button" id="final_confirm_btn-${num}" finish_generation="true" confirmed="false"><i class="fas fa-check-double" aria-hidden="true" id="final_confirm_btn_icon-${num}" title="Confirm" ></i></button--></div>
    </div>
  `
  let template_plus = 
      `
        <div id="${num}_plus">
          <button class="convline-plus-btn" type="button" id="convline_plus-${num}"><i class="fas fa-plus-circle"
              aria-hidden="true" id="convline_plus_icon-${num}" title="Add" style="color: aliceblue;"></i></button>
        </div>
      `
  let template_plus_script = 
      `
          
            $("#convline_plus-${num}").click(function () {
              let itm = $(this);
              let itm_idx = itm.attr('id');
              addConvlineWords(itm_idx);
            })
          `
  let template_confirm_all = 
      `
        <div id="${num}_confirmAll">
          <button class="convline-confirm-all-btn" style="padding: 0 0 0 15" type="button" id="confirm_all_btn-${num}" confirmall="false" finish_generation="true"><i class="fas fa-redo"
              aria-hidden="true" id="confirm_all_btn_icon-${num}" title="Confirm and generate response" ></i></button>
        </div>
      `;
  let template_confirm_all_script = 
      `
          
            $("#confirm_all_btn-${num}").click(function () {
              let itm = $(this);
              let itm_idx = itm.attr('id');
              regenerate(itm_idx);
            })
          
      `;
      let template_final_confirm_btn = 
      `
      </div> <button class="final-confirm-btn" type="button" id="final_confirm_btn-${num}" finish_generation="true" confirmed="false"><i class="fas fa-check-double" aria-hidden="true" id="final_confirm_btn_icon-${num}" title="Confirm" ></i></button>
      `
  let template_final_confirm_script = 
      `
      $("#final_confirm_btn-${num}").click(function () {
        let itm = $(this);
        let itm_idx = itm.attr("id");
        let flag = document.getElementById("main-message-box").getAttribute("firstacceptclick")
        if (flag == "true" && convline_status=="on") {
          if (confirm("Warning: once click on Confirm or SEND button, you cannot change the ConvLine word anymore! (This message won't show later. )")) {
            document.getElementById("main-message-box").setAttribute("firstacceptclick", "false");
            ConfirmAll(itm_idx);
          } else {
            document.getElementById("main-message-box").setAttribute("firstacceptclick", "false");
          }
        } else {
          ConfirmAll(itm_idx);
        }
      });
      `
      let template_regenerate_keywords = 
      `
      <div id="${num}_keywords_regenerate">
        <button class="convline-confirm-all-btn" type="button" id="keywords_regenerate_btn-${num}" confirmall="false" finish_generation="true"><i class="fas fa-redo"
            aria-hidden="true" id="keywords_regenerate_btn_icon-${num}" title="Confirm and generate response" ></i></button>
      </div>
      `
  let template_regenerate_keywords_script = 
      `
      $("#keywords_regenerate_btn-${num}").click(function () {
        let itm = $(this);
        let itm_idx = itm.attr('id');
        keywords_regenerate(itm_idx);
      })
      `
  $(template_frame).appendTo($(".mCSB_container")).addClass("new");
  
  $(template_plus).appendTo($(`#convline_${num}`));
  $(template_confirm_all).appendTo($(`#${num}_responseline`));
  let childNode1 = document.createElement('script');
  childNode1.innerHTML = template_plus_script;
  document.getElementById(`${num}_plus`).appendChild(childNode1);
  let childNode2 = document.createElement('script');
  childNode2.innerHTML = template_confirm_all_script;
  document.getElementById(`${num}_confirmAll`).appendChild(childNode2);
  $(template_final_confirm_btn).appendTo($(`#${num}_responseline`));
  let childNode3 = document.createElement('script');
  childNode3.innerHTML = template_final_confirm_script;
  document.getElementById(`${num}_responseline`).appendChild(childNode3);
  $(template_regenerate_keywords).appendTo($(`#convline_${num}`));
  let childNode4 = document.createElement('script');
  childNode4.innerHTML = template_regenerate_keywords_script;
  document.getElementById(`${num}_keywords_regenerate`).appendChild(childNode4);
  if (convlines!=null){
    if (convlines.length>0){
      for (i=0;i<convlines.length;i++) {
        addConvlineWords(`convline_plus-${num}`, convlines[i], true);
      }
    }
  }
  if (convline_status=="off") {
    $(`#convline_${num}`).addClass('close-convline')
    $(`#final_confirm_btn-${num}`).addClass('close-final-btn')
    $(`#confirm_all_btn-${num}`).addClass('close-final-btn')
  }
}

function keywords_regenerate(idx) {
  let num = idx.split('-')[1];
  disable_convline('keywords-regeneration', num);
  let nodes = document.getElementById("mCSB_1_container").lastChild.querySelectorAll(".convline-word");
  let msg_nodes = document.getElementsByClassName("message")
  let history = []
  for (i=0;i<msg_nodes.length;i++){
    history.push(msg_nodes[i].querySelector(".message-text").innerHTML)
  }
  sendData = {
    "history": history,
    //"keywords": convline_words,
    "temperature": document.getElementById('temperature').value,
    "top-k": document.getElementById('top-k').value,
    "top-p": document.getElementById('top-p').value
  }
  sendData = JSON.stringify(sendData);
  //remove original convlines
  let past_convline = document.getElementById(`confirm_all_btn-${num}`).parentNode.parentNode.parentNode.querySelectorAll('.convline-sample');
  if (past_convline.length>0) {
    for (i=0;i<past_convline.length;i++) {
      past_convline[i].remove();
    }
  }
  // generate animate for message waiting
  let el1 = document.getElementById(`convline_plus-${num}`).parentNode;
  el1.insertAdjacentHTML('beforebegin','<div class="message1 loading new" id="convlines_animation"><span></span></div>')
  let text_node = document.getElementById('convlines_animation');
  $.ajax({
    url:'/regenerate-keywords/',
    method: 'POST',
    dataType: 'json',
    contentType: 'application/json;charset=utf-8',
    data: sendData,
    async: true,
    timeout: 0,
    crossDomain: true,
    success: function (data) {
      console.log('receive api request')
      console.log(data)
    },
    error: function(jqxhr, status, exception) {
      console.log('Exception:', JSON.stringify(exception, null, 2));
      console.log('fail to get request response')
      $(".notify1").remove("active")
      $("#notifyType1").remove("sending")
      $(".notify1").addClass("error")
      $(".notify1").addClass("active")
      $(".notify1").attr("color","red")
      $("#MSG").removeAttr('disabled')
      $("#msg-btn").css("background-color", "#248A52")
      $("#msg-btn").removeClass("not-changable")
      $("#MSG").attr("placeholder",'type message...')
      $(".parameter-group").removeClass("not-allow")
      $(".form-control-range").removeClass("not-changable")
      $(".submission-group").removeClass("not-allow")
      $(".submit-button").removeClass("not-changable")
      $("#convline-onoff").removeClass("not-changable")
      setTimeout(function(){
        $(".notify1").removeClass("active");
        $("#notifyType").removeClass("error");
        $(".notify1").setAttribute("color","white")
      },10000);
    },
    beforeSend: function() {
      $(".notify1").remove("active")
      $("#notifyType1").addClass("sending")
      $(".notify1").addClass("active")
      $("#MSG").attr('disabled','disabled');
      $("#msg-btn").css("background-color", "gray")
      $("#msg-btn").addClass("not-changable")
      $("#MSG").attr("placeholder",'🚫')
      //$('<div class="message loading new"><figure class="avatar"><img src="static/css/pluslab-icon.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
      $(".parameter-group").addClass("not-allow")
      $(".form-control-range").addClass("not-changable")
      $(".submission-group").addClass("not-allow")
      $(".submit-button").addClass("not-changable")
      $("#convline-onoff").addClass("not-changable")
      updateScrollbar();
    }, 
    complete: function(data) {
      $(".notify1").remove("active")
      $("#notifyType1").remove("sending")
      $(".notify1").addClass("getMsg")
      $(".notify1").addClass("active")
      $("#MSG").removeAttr('disabled')
      $("#msg-btn").css("background-color", "#248A52")
      $("#msg-btn").removeClass("not-changable")
      $("#MSG").attr("placeholder",'type message...')
      $(".parameter-group").removeClass("not-allow")
      $(".form-control-range").removeClass("not-changable") 
      $(".submission-group").removeClass("not-allow")
      $(".submit-button").removeClass("not-changable")
      $("#convline-onoff").removeClass("not-changable")
      setTimeout(function(){
        $(".notify1").removeClass("active");
        $("#notifyType").removeClass("sending");
      },1000);
      //$('.message.loading').remove();
      //if (typeof data === 'string'){
      //  data = JSON.parse(data);
      //  console.log(22)
      //}
      let responses = data.responseJSON["responses"];
      let convlines = data.responseJSON["keywords"].split("#");
      for (i=0;i<convlines.length;i++) {
        convlines[i] = convlines[i].trim();
      }
      console.log(data.responseJSON)
      console.log(responses)
      console.log(convlines)
      document.getElementById(`${num}_plus`).hidden=false;
      document.getElementById(`final_confirm_btn_icon-${num}`).style.color = "aliceblue";
      document.getElementById(`final_confirm_btn-${num}`).style.pointerEvents = "auto";
      document.getElementById(`confirm_all_btn_icon-${num}`).style.color = "aliceblue";
      document.getElementById(`confirm_all_btn-${num}`).style.pointerEvents = "auto";
      document.getElementById(`keywords_regenerate_btn-${num}`).hidden=false;
      //document.getElementById(`confirm_all_btn-${num}`).setAttribute("confirmall", "false");
      //document.getElementById(`confirm_all_btn-${num}`).hidden=false;
      text_node.remove();
      for (i=0;i<convlines.length;i++) {
        addConvlineWords(`convline_plus-${num}`, convlines[i], true);
      }
      
      let idx = parseInt(num)
      message_list["chatbot"][idx].push([convlines, '', [document.getElementById('temperature').value, document.getElementById('top-k').value,  document.getElementById('top-p').value]])
    }
  });
}




