<template>
  <main class="page-datasets">
      <SvgContainer name="arrow-right-active-fill" />
      <SvgContainer name="arrow-right-outline" />
      <SvgContainer name="arrow-right-fill" />
      <SvgContainer name="notification-outline" />
      <SvgContainer name="help-circle" />
      <SvgContainer name="google-drive" />
      <SvgContainer name="file-outline" />
      <SvgContainer name="folder" />
      <SvgContainer name="folder-add" />
      <SvgContainer name="folder-save" />
      <SvgContainer name="folder-validate" />
      <SvgContainer name="folder-remove" />
      <SvgContainer name="sloy-start-add" />
      <SvgContainer name="sloy-middle-add" />
      <SvgContainer name="sloy-end-add" />
  </main>
</template>

<script>
import SvgContainer from "@/components/app/SvgContainer"
export default {
  name: 'Datasets',
  components: {
      SvgContainer
  }
};
</script>