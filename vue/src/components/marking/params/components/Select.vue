<template>
  <div class="t-field">
    <label class="t-field__label">{{ label }}</label>
    <input style="display: none" :name="parse" :value="select" />
    <at-select
      v-model="select"
      :class="['t-field__select', { 't-field__error': error }]"
      clearable
      size="small"
      @on-change="change"
      @click="cleanError"
      :disabled="disabled"
      width="180px"
    >
      <at-option v-for="({ label, value }, key) in items" :key="'item_' + key" :value="value" :title="label">
        {{ label }}
      </at-option>
    </at-select>
  </div>
</template>

<script>
export default {
  name: 't-select',
  props: {
    label: {
      type: String,
      default: 'Label',
    },
    type: {
      type: String,
      default: '',
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
    disabled: Boolean,
    error: String,
  },
  data: () => ({
    select: '',
  }),
  computed: {
    items() {
      if (Array.isArray(this.lists)) {
        return this.lists.map(i => {
          return i || '';
        });
      } else {
        return Object.keys(this.lists);
      }
    },
  },
  methods: {
    cleanError() {
      if (this.error) {
        this.$emit('cleanError')
      }
    },
    change(value) {
      this.$emit('input', value);
      this.$emit('change', { name: this.name, value });

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
    padding: 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    // white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    height: 24px;
    font-size: 0.75rem;
    max-width: 180px;
  }
  &__select {
    flex: 0 0 180px;
  }
  &__error {
    border-color: #b53b3b;
  }
}
</style>
