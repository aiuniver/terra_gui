<template>
  <div class="t-field">
    <label class="t-field__label">{{ label }}</label>
    <input style="display: none" :name="parse" :value="select" />
    <at-select
      class="t-field__select"
      v-model="select"
      size="small"
      style="width: 100px"
      @on-change="change"
      :disabled="disabled"
    >
      <at-option
        v-for="({ label, value }, key) in items"
        :key="'item_' + key"
        :value="value"
      >
        {{ label }}
      </at-option>
    </at-select>
  </div>
</template>

<script>
// import { bus } from '@/main'
export default {
  name: "TSelect",
  props: {
    label: {
      type: String,
      default: "Label",
    },
    type: {
      type: String,
      default: "",
    },
    value: {
      type: [String, Number],
    },
    name: {
      type: String,
    },
    parse: {
      type: String,
    },
    lists: {
      type: [Array, Object],
    },
    disabled: Boolean
  },
  data: () => ({
    select: "",
  }),
  computed: {
    items() {
      if (Array.isArray(this.lists)) {
        return this.lists.map((i) => {
          return i || "";
        });
      } else {
        return Object.keys(this.lists);
      }
    },
  },
  methods: {
    change(value) {
      this.$emit("input", value);
      // this.$emit("change", { name: this.name, value });
      console.log(value)
      console.log((value === '__null__'))
      this.$emit('change', { name: this.name, value: (value === '__null__') ? null : value });

      // bus.$emit("change", e);
    },
  },
  created() {
    this.select = this.value;

    // console.log('created', this.select)
    // console.log('created', this.name);
    // bus.$on("change", () => {
    //   console.log(this.name, 'data');
    // });
  },
  destroyed() {
    // bus.$off()
    // console.log('destroyed', this.name);
  },
};
</script>


<style lang="scss" scoped>
.t-field {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;
  &__label {
    width: 150px;
    max-width: 130px;
    padding: 0 10px 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    height: 22px;
    font-size: 0.75rem;
    max-width: 109px;
    width: 109px;
  }
  &__select {
    flex: 0 0 100px;
  }
}
</style>
