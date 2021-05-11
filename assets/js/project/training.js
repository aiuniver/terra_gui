"use strict";


(($) => {


    $.fn.extend({


        TrainingParams: function() {

            if (!this.length) return this;

            let _field_optimizer = $("#field_form-optimazer");

            _field_optimizer.selectmenu({
                change:(event) => {
                    $(event.target).trigger("change");
                }
            }).bind("change", (event) => {
                let name = $(event.currentTarget).val(),
                    optimizer = window.TerraProject.training.optimizer;
                console.log(optimizer.params);
                // window.TerraProject.set_optimizer();
            }).trigger("change");

            this.bind("submit", (event) => {
                event.preventDefault();
                console.log($(event.currentTarget).serializeObject());
            });

            return this;

        }


    })


    $(() => {

        $(".project-training-properties > .wrapper > .params > .params-container").TrainingParams();

    });


})(jQuery);
