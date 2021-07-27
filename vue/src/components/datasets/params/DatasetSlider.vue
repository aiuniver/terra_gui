<template>
  <div class="field-form field-inline align-center row">
    <div class="col-7 label">Train + val + test</div>
    <div class="col-7"><Range @set="change" /></div>
    <div class="col-10 input">
      <input type="number" name="parameters[train_part]" v-model="min"/>
      <input type="number" name="parameters[val_part]" v-model="delta"/>
      <input type="number" name="parameters[test_part]" v-model="max"/>
    </div>
  </div>
</template>

<script>
import Range from "@/components/forms/Range.vue";
export default {
  props: {},
  components: {
    Range,
  },
  data: () => ({
    min: 10,
    delta: 10,
    max: 30
  }),
  methods: {
    change({ min, max }) {
      console.log( min, max )
      this.min = min
      this.delta = max - min
      this.max = max
    }
  }
};
</script>

<style scoped>
.input input {
  margin-right: 1px;
  max-width: 45px;
  height: 22px;
  font-size: 0.75rem;
}
.label {
  font-size: 0.75rem;
  margin-bottom: 5px;
}
</style>