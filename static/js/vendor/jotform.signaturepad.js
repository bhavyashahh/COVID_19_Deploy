jQuery(document).ready(function() {
  var debug = document.get.debug === '1';

  jQuery(".pad").each(function(idx, el) {
    var pad = jQuery(el);
    var qid = pad.data('id');
    var pwidth = pad.data("width");
    var pheight = pad.data("height");
    var signatureLine = pad.data("signature-line") == "1";

    // if main wrapper size is not zero use it
    // we maybe on the mobile version of the form
    var wrapperEl = jQuery('#signature_pad_' + qid);
    var wrapperWidth = wrapperEl.width();
    if (JotForm.isVisible(wrapperEl[0]) && wrapperWidth > 0) {
      pwidth = wrapperWidth;
      debug && console.log('Wrapper is visible and has a width', wrapperEl, pwidth);
    }

    // set widths to pad and signature canvas
    pad.width(pwidth).height(pheight);
    pad.jSignature({
      width: pwidth,
      height: pheight,
      signatureLine: signatureLine
    });

    // debug signature pad
    debug && console.log('Pad info', qid, pad, pwidth, pheight);

    // bind changes - emits 'change' event immediately after a stroke
    pad.bind('change', function() {
      var thispad = jQuery(this);
      var qid = thispad.attr('data-id');
      if (thispad && typeof thispad.jSignature !== 'undefined' && thispad.jSignature('getData', 'base30')[1].length > 0) {
        var sigdata = thispad.jSignature('getData');
        jQuery("#input_" + qid).val(sigdata);
        JotForm.triggerWidgetCondition(qid);
      }
    });

    // a prototype event that will resize canvas onresize
    el.observe('on:sigresize', function(event) {
      var qid = el.readAttribute('data-id');
      var newWrapperWidth = jQuery('#signature_pad_' + qid).width();

      // only resize if the new wrapper width is not zero
      if (newWrapperWidth > 0) {
        var newwidth = newWrapperWidth + 'px';
        el.setStyle({
            'width': newwidth
          })
          .select('canvas.jSignature').first()
          .setStyle({
            'width': newwidth
          })
          .writeAttribute('width', newwidth);

        // use jquery to reset jSignature
        pad.jSignature("reset");

        debug && console.log('Pad on:sigresize event', qid, pad, newwidth);
      }
    });
  });

  jQuery(".clear-pad").bind('click keypress', function(e) {
    if (e.type === 'click' || e.keyCode === 13) {
      var pad = jQuery(this).prev().find(".pad");
      pad.jSignature("reset");

      // clear input field as well
      var qid = pad.attr('data-id');
      jQuery("#input_" + qid).val('');
      JotForm.triggerWidgetCondition(qid);
    }
  });

  jQuery(".jotform-form").submit(function(e) {
    jQuery(".pad").each(function(idx, el) {
      var pad = jQuery(el);
      if (!pad.hasClass('edit-signature') && pad.jSignature('getData', 'base30')[1].length > 0) {
        var id = pad.attr("data-id");
        jQuery("#input_" + id).val(pad.jSignature('getData'));
      }
    });
  });

  //@diki
  //edit mode
  if (JotForm.isEditMode() || typeof document.get.session !== 'undefined' || document.location.href.match(/\/edit\//)) {
    jQuery(".jotform-form").on("click keypress", ".edit-signature-pad", function(e) {
      if (e.type === 'click' || e.keyCode === 13) {
        // get pad and the pad id
        var sigId = jQuery(this).attr('data-id');
        var pad = jQuery('.pad#sig_pad_' + sigId);

        // if there's a sig image and want to clear it
        if (jQuery('img.signature-image-' + sigId).length > 0) {
          // show the pad and hide flag class
          pad.removeClass('edit-signature').show();

          // remove value from the input
          jQuery('#input_' + sigId).val('');

          // remove current signature image
          jQuery('img.signature-image-' + sigId).remove();
        } else {
          // reset pad
          pad.jSignature("reset");
        }
      }
    });

  }
});
