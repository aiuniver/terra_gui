<template>
  <div class="field-form field-inline field-reverse">
    <label @click="checked = !checked">{{ label }}</label>
    <div class="checkout-switch">
      <input
        v-model="checked"
        :checked="value ? 'checked' : ''"
        :type="type"
        :value="checked"
        :name="parse"
        @change="change(value)"
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
      type: [String, Number, Boolean],
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
    change() {
      console.log(this.name, this.checked);
      bus.$emit("change", { event: this.name, value: this.checked });
    },
  },
  created() {
    this.checked = this.value;
    if (this.event.length) {
      console.log("created", this.name);
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

