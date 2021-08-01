<template>
  <div class="field-form field-inline field-reverse">
    <label :for="parse">{{ label }}</label>
    <div class="checkout-switch">
      <input
        :id="parse"
        :checked="checked ? 'checked' : ''"
        :type="type"
        :value="checked"
        :name="parse"
        @change="change"
      />
      <span class="switcher"></span>
    </div>
  </div>
</template>

<script>
import { bus } from "@/main";
export default {
  props: {
    label: {
      type: String,
      default: "Label",
    },
    type: {
      type: String,
      default: "text",
    },
    value: {
      type: [Boolean],
    },
    name: {
      type: String,
    },
    parse: {
      type: String,
    },
    event: {
      type: Array,
      default: () => [],
    },
  },
  data: () => ({
    checked: null,
  }),
  methods: {
    change(e) {
      // console.log(e);
      const value = e.target.checked
      this.$emit("change", { name: this.name, value });
      bus.$emit("change", { event: this.name, value });
    },
  },
  created() {
    this.checked = this.value;
    // console.log('created ' + this.name, this.checked)
    if (this.event.length) {
      // console.log("created", this.name);
      bus.$on("change", ({ event }) => {
        if (this.event.includes(event)) {
          this.checked = false;
        }
      });
    }
  },
  destroyed() {
    if (this.event.length) {
      bus.$off();
      console.log("destroyed", this.name);
    }
  },
};
</script>

