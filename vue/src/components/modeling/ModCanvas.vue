<template>
<div class="board">
    <div class="wrapper">
        <div class="canvas-container">
            <div class="canvas">
              <VueFlowchart></VueFlowchart>
            </div>
            <ul class="zoom">
                <li class="inc" id="zoom-inc" data-type="inc"><span class="icon-zoom-inc"></span></li>
                <li class="res" id="zoom-reset" data-type="reset"><span class="icon-zoom-reset"></span></li>
                <li class="dec" id="zoom-dec" data-type="dec"><span class="icon-zoom-dec"></span></li>
            </ul>
        </div>
    </div>
</div>
</template>

<script>
import VueFlowchart from "@/components/modeling/flowchart/SimpleFlowchart";
export default {
  name: "ModCanvas",
  components:{
    VueFlowchart,
  }
}
</script>

<style scoped>

</style>