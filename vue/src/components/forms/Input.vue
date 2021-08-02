<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" :for="parse">{{ label }}</label>
    <input
      v-model="input"
      class="t-field__input"
      :id="parse"
      :type="type"
      :name="parse"
      :value="value"
      @blur="change"
    />
  </div>
</template>

<script>
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
      type: [String, Number],
    },
    parse: String,
    name: String,
    inline: Boolean,
  },
  data: () => ({
    isChange: false
  }),
  computed: {
    input: {
      set(value) {
        this.$emit('input', value)
        this.isChange = true
      },
      get() {
        return this.value
      }
    }
  },
  methods: {
    change(e) {
      if (this.isChange) {
        let value = e.target.value
        value = this.type === 'number' ? +value : value
        this.$emit('change', { name: this.name, value } )
        this.isChange = false
      }
    }
  }
};
</script>

<style lang="scss" scoped>
.t-field {
  margin-bottom: 20px;
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
    color: #fff;
    border-color: #6C7883;
    background: #242F3D;
    height: 42px;
    padding: 0 10px;
    font-size: .875rem;
    font-weight: 400;
    border-radius: 4px;
    transition: border-color .3s ease-in-out, opacity .3s ease-in-out;
    &:focus{
      border-color: #fff;
    }
  }
}
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;
  > label {
    width: auto;
    padding: 0 20px 0 10px;
    text-align: left;
    color: #A7BED3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: .75rem;
  }
  > input {
    height: 22px;
    font-size: 12px;
    line-height: 24px;
    width: 109px;

  }
}
</style>
