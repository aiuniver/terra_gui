<template>
  <div class="two-inputs">
    <div class="two-inputs__inputs">
      <label for="field_form-num_links[inputs]">Кол-во <b>входов</b></label
      ><input id="field_form-num_links[inputs]" type="number" />
    </div>
    <div class="two-inputs__outputs">
      <label for="field_form-num_links[outputs]">Кол-во <b>выходов</b></label
      ><input id="field_form-num_links[outputs]" type="number" />
    </div>
  </div>
</template>

<script>
export default {
  name: "DatasetTwoInput",
  props: {},
  data: () => ({
    select: "GoogleDrive",
  }),
  methods: {
    click(mode) {
      this.select = mode;
      this.items = this.items.map((item) => {
        return { ...item, active: item.mode === mode };
      });
    },
  },
};
</script>

<style lang="scss" scoped>
.two-inputs {
  display: flex;
  &__inputs,
  &__outputs {
    display: flex;
    align-items: center;
    > label {
      order: 1;
      width: auto;
      padding: 0 20px 0 10px;
      text-align: left;
      color: #a7bed3;
      display: block;
      // margin: 0 0 10px 0;
      line-height: 1.25;
      font-size: 0.75rem;
      user-select: none;
      > b {
        color: #fff;
      }
    }
    > input {
      max-width: 45px;
      height: 22px;
      font-size: 0.75rem;
      color: #fff;
      border-color: #6c7883;
      background: #242f3d;
    }
  }
}
</style>