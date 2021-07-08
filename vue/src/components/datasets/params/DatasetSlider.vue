<template>
  <div class="field-form field-inline row">
    <div><label>Train + val + test</label></div>
    <div><Range /></div>
    <div class="input">
      <input type="number" name="parameters[train_part]" />
      <input type="number" name="parameters[val_part]" />
      <input type="number" name="parameters[test_part]" />
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
};
</script>

<style scoped>
.input input {
  max-width: 45px;
  height: 22px;
  font-size: 0.75rem;
}
</style>