<template>
  <scrollbar>
    <div class="block-header">
      <CsvTable/>
      <CardFile :title="'BMW'" :color="'#FFB054'" />
      <CardFile :title="'AUDI'" :color="'#89D764'" />
      <CardFile :title="'Ferrari'" :color="'#8E51F2'" />

    </div>
  </scrollbar>
</template>

<script>
import CardFile from "../components/card/CardFile.vue";
import CsvTable from "../components/forms/CsvTable";
export default {
  name: "BlockHeader",
  components: {
    CardFile,
    CsvTable
  },
};
</script>

<style lang="scss" scoped>
.block-header {
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;
  padding: 10px;
}
</style>