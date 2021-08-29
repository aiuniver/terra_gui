<template>
  <div :class="'t-field-inline'">
    <label class="t-field-inline__label" @click="click">
      {{ label }}
    </label>
    <div class="t-field-inline__input" slot="input">
      <t-checkbox class="input" />
    </div>
  </div>
</template>

<script>
export default {
  name: 't-field-inline',
  props: {
    label: {
      type: String,
      default: 'Label',
    },
  },
  data: () => ({}),
  computed: {},
  methods: {
    click() {
      console.log(this.$el.getElementsByTagName('input')[0].focus())
    }
  },
};
</script>

<style lang="scss" scoped>
.input {
  color: #fff;
  background: #242f3d;
  height: 100%;
  width: 100%;
  padding: 0 10px;
  font-size: 0.875rem;
  font-weight: 400;
  border-radius: 4px;
  border: 1px solid #6c7883;
  transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
}
.t-field-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  // align-items: center;
  &__label {
    width: 150px;
    max-width: 130px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1;
    font-size: 0.75rem;
  }
  &__input {
    height: 24px;
    font-size: 12px;
    line-height: 24px;
    width: 100px;
  }
}
</style>
